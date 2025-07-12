import os
import argparse
from typing import Tuple

import yaml
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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
    save_path: str = "data.bin",
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
    np_token_ids.tofile(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(config["DATA"]["DATA_PATH"]):
        create_bin(
            config["DATA"]["HF_DATA_PATH"],
            config["DATA"]["TOKENIZER_ID"],
            config["DATA"]["SEQ_LEN"],
            save_path=config["DATA"]["DATA_PATH"],
        )

    dataset = NextTokenDataset(
        data_path=config["DATA"]["DATA_PATH"], seq_len=config["DATA"]["SEQ_LEN"]
    )

    x, y = dataset[0]
    print(x.shape, y.shape)
