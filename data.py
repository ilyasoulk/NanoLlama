import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


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
    data_path = "roneneldan/TinyStories"
    tokenizer_id = "HuggingFaceTB/SmolLM2-135M"
    seq_len = 128

    create_bin(data_path, tokenizer_id, seq_len)
