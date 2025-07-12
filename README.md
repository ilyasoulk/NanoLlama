# Nano Llama

Single file implementation of llama3 models with pre-training script


# TODO

- [x] base architecture in pytorch
- [x] load hf weights into my implem
- [x] tests
- [x] auto-regressive dataset
- [ ] add wandb
- [ ] simple pretrain run (cosine scheduler, adamw, grad clip, mix precision)
- [ ] batch scheduling, custom dataloader ? control amount of token per batch...
- [ ] WSD scheduler
- [ ] SGD with batch size 1
- [ ] muon ?

# Test

```sh
# Install dependencies
git clone https://github.com/ilyasoulk/NanoLlama.git
cd NanoLlama
uv sync
source .venv/bin/activate
# Run SmolLM2-135M
python llama.py --model_id HuggingFaceTB/SmolLM2-135M
```
