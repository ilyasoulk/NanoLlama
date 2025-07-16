# Nano Llama

Single file implementation of llama3 models with pre-training script


# TODO

- [x] base architecture in pytorch
- [x] load hf weights into my implem
- [x] tests
- [x] auto-regressive dataset
- [x] simple pretrain script
- [x] fix huge abnormal loss (~120) should expect around log(vocab_size=49k) = ~10.8
- [x] add wandb
- [ ] what architecture ? how many params ?
- [ ] try multiple optimizers sgd vs adamw vs muon, mixed precision
- [ ] batch scheduling, custom dataloader ? control amount of token per batch...
- [ ] WSD scheduler

# Test

```sh
# Install dependencies
git clone https://github.com/ilyasoulk/NanoLlama.git
cd NanoLlama
uv sync
source .venv/bin/activate

# Run SmolLM2-135M
python llama.py --model_id HuggingFaceTB/SmolLM2-135M

# Run pre-training
python train.py --config_file configs/train/setup.yaml
```
