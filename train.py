import os
import time
import random
import argparse
from itertools import cycle
from tqdm import tqdm
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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader


class NextTokenDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r").reshape(-1, seq_len)
        print(f"Number of tokens : {(self.data.size / 1e9):.4f}M")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data[index].astype(np.int64)

        x = torch.from_numpy(row[:-1])
        y = torch.from_numpy(row[1:])
        return x, y


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_bin(
    data_path: str,
    tokenizer_id: str,
    seq_len: int,
    num_samples: int | None = 10_000,
    save_path: str = "./data",
) -> None:
    train_dataset = load_dataset(data_path, split="train", streaming=True)
    test_dataset = load_dataset(data_path, split="validation", streaming=True)

    if num_samples is not None:
        train_dataset = list(train_dataset.take(num_samples))  # type: ignore
        test_dataset = list(test_dataset)
    else:
        train_dataset = list(train_dataset)
        test_dataset = list(test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<bos>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<eos>"})

    bos, eos = tokenizer.bos_token, tokenizer.eos_token

    def tokenize(dataset):
        corpus = "".join([f"{bos} {d['text']} {eos}" for d in dataset])
        tokens = tokenizer(corpus)["input_ids"]
        num_tokens = len(tokens)
        num_seq, r = divmod(num_tokens, seq_len)
        return np.array(tokens[:-r], dtype=np.uint16).reshape(num_seq, seq_len)

    train_data = tokenize(train_dataset)
    test_data = tokenize(test_dataset)

    os.makedirs(save_path, exist_ok=True)
    train_data.tofile(os.path.join(save_path, "train.bin"))
    test_data.tofile(os.path.join(save_path, "test.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config["device"] == "cuda":
        torch.cuda.manual_seed(seed)

    seq_len = config["data"]["seq_len"]

    if not os.path.exists(config["data"]["data_path"]):
        create_bin(
            config["data"]["hf_data_path"],
            config["data"]["tokenizer_id"],
            seq_len,
            save_path=config["data"]["data_path"],
            num_samples=None,
        )

    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = NextTokenDataset(
        data_path=config["data"]["train_data_path"], seq_len=config["data"]["seq_len"]
    )
    test_dataset = NextTokenDataset(
        data_path=config["data"]["test_data_path"], seq_len=config["data"]["seq_len"]
    )
    batch_size = config["optim"]["batch_size"]
    grad_accum_steps = config["optim"]["grad_accum_steps"]
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        worker_init_fn=seed_worker,
        generator=g,
    )

    device = torch.device(config["device"])
    llama_config = LlamaConfig(**config["llama_config"])
    model = Llama(llama_config).to(device)
    num_params = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters : {(num_params / 1e6):.2f}M")

    lr = float(config["optim"]["lr"])
    if "adamw" in config["optim"]:
        beta1, beta2 = (
            config["optim"]["adamw"]["beta1"],
            config["optim"]["adamw"]["beta2"],
        )
        eps = float(config["optim"]["adamw"]["eps"])
        weight_decay = config["optim"]["adamw"]["weight_decay"]
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, config["optim"]["t_max"]
    # )

    max_norm = config["optim"]["max_norm"]

    wandb.init(
        project="Llama3",
        config=config,
    )

    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

    num_tokens = 0
    steps = config["optim"]["steps"]
    train_iter = cycle(train_loader)
    for step in range(0, steps):
        acc_loss = 0.0
        start = time.time()
        for i in range(grad_accum_steps):
            x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss /= grad_accum_steps
            acc_loss += loss.item()
            loss.backward()

            num_tokens += x.numel()

        clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        end = time.time()
        optimizer.zero_grad()
        # scheduler.step()

        duration = end - start
        tokens_per_second = (batch_size * grad_accum_steps * seq_len) / duration

        print(
            f"Steps: {step} | train loss : {acc_loss:.4f} | tokens/s : {tokens_per_second:.2f}"
        )
        wandb.log({
            "steps": step,
            "train_loss": acc_loss,
            "tokens": num_tokens,
        })

        acc_loss = 0.0
        start = time.time()

        if step % config["test_freq"] == 0:
            test_loss = 0.0
            model.eval()  # type:ignore
            for i, (x, y) in tqdm(enumerate(test_loader), desc="Testing..."):
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

                test_loss += loss.item()

            avg_test_loss = test_loss / len(test_loader)
            print(f"Step: {step} | test_loss: {avg_test_loss}")
            wandb.log({
                "step": step,
                "test_loss": avg_test_loss,
                "tokens": num_tokens,
            })
