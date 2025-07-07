# Nano Llama

Single file implementation of llama3 models with pre-training script


# TODO

- [x] Base architecture in pytorch
- [x] Load hf weights into my implem
- [x] Tests
- [ ] Training script
- [ ] Maybe FSDP

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
