import os
import time
import argparse
from typing import Tuple


from llama import Llama, LlamaConfig

import yaml
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader


class NextTokenDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r").reshape(-1, seq_len)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data[index].astype(np.int64)

        x = torch.from_numpy(row[:-1])
        y = torch.from_numpy(row[1:])
        return x, y


def create_bin(
    data_path: str,
    tokenizer_id: str,
    seq_len: int,
    num_samples: int = 10_000,
    save_path: str = "./data",
    split: float = 0.8,
) -> None:
    dataset = load_dataset(data_path, split="train", streaming=True)
    data = list(dataset.take(num_samples))  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<bos>"})

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<eos>"})

    bos = tokenizer.bos_token
    eos = tokenizer.eos_token

    corpus = "".join([f"{bos} {d['text']} {eos}" for d in data])

    outputs = tokenizer(corpus)
    token_ids = outputs["input_ids"]

    num_tokens = len(token_ids)
    num_seq, r = divmod(num_tokens, seq_len)

    np_token_ids = np.array(token_ids[:-r], dtype=np.uint16).reshape(num_seq, seq_len)
    split_idx = int(split * num_seq)
    train_data = np_token_ids[:split_idx]
    test_data = np_token_ids[split_idx:]

    os.makedirs(save_path, exist_ok=True)
    train_data.tofile(os.path.join(save_path, "train.bin"))
    test_data.tofile(os.path.join(save_path, "test.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(config["data"]["data_path"]):
        create_bin(
            config["data"]["hf_data_path"],
            config["data"]["tokenizer_id"],
            config["data"]["seq_len"],
            save_path=config["data"]["data_path"],
        )

    train_dataset = NextTokenDataset(
        data_path=config["data"]["train_data_path"], seq_len=config["data"]["seq_len"]
    )
    test_dataset = NextTokenDataset(
        data_path=config["data"]["test_data_path"], seq_len=config["data"]["seq_len"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["optim"]["batch_size"], shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config["optim"]["batch_size"])

    device = torch.device(config["device"])
    llama_config = LlamaConfig(**config["llama_config"])
    model = Llama(llama_config).to(device)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters : {num_params}")

    optimizer = torch.optim.SGD(model.parameters(), config["optim"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["optim"]["t_max"]
    )

    epochs = config["optim"]["epochs"]

    tokens_per_batch = config["optim"]["batch_size"] * config["data"]["seq_len"]
    num_batches = len(train_loader)

    wandb.init(
        project="Llama3",
        config=config,
    )

    model = torch.compile(model)

    num_tokens = 0
    for epoch in range(epochs):
        train_loss = 0.0
        start = time.time()
        for i, (x, y) in enumerate(train_loader):
            if i > 10:  # temp, for testing
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_tokens += num_batches * tokens_per_batch

        end = time.time()
        duration = end - start
        avg_train_loss = train_loss / num_batches
        tokens_per_second = (tokens_per_batch * num_batches) / duration
        print(
            f"Epoch: {epoch}, train loss : {avg_train_loss}, tokens/s : {tokens_per_second}"
        )
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "tokens": num_tokens})
