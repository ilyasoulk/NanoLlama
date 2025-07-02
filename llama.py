import yaml
import argparse
from dataclasses import dataclass


import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LlamaConfig:
    d_model: int
    vocab_size: int
    kv_heads: int
    q_heads: int
    d_head: int
    num_layers: int
    seq_len: int
    rms_norm_eps: float
    batch_size: int
    causal: bool = False
    do_flash: bool = False


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Swiglu(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2 * self.d_model
        a = x[..., : self.d_model]
        b = x[..., self.d_model :]
        return a * (b * torch.sigmoid(b))


class LlamaMLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.up_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.down_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.act_fn = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x_up)


class ROPE(nn.Module):
    def __init__(self, d_head: int, base: float = 10_000.0) -> None:
        super().__init__()
        freq = 1.0 / (base) ** (torch.arange(0, d_head, 2) / d_head)  # (D / 2,)
        self.register_buffer("freq", freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) % 2 == 0  # (B, S, N_head, D_head)
        B, S, N_head, D_head = x.shape
        x_complex = x.view(
            B, S, N_head, D_head // 2, 2
        )  # (B, S, N_head, D_head // 2, 2)

        pos = torch.arange(0, S, device=x.device, dtype=x.dtype)  # (S,)
        theta = torch.outer(pos, self.freq)  # (S, D/2)
        cos, sin = (
            theta.cos()[None, :, None, :],
            theta.sin()[None, :, None, :],
        )  # (1, S, 1, 1, D/2)

        x1, x2 = x_complex.unbind(-1)
        out = torch.stack(
            (x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
        )  # (B, S, N_head, D_head // 2, 2)

        return out.flatten(-2)  # (B, S, N_head, D_head)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x**2, keepdim=True, dim=-1).sqrt()
        return (x / (rms + self.eps)) * self.gain

    def __repr__(self):
        return f"{self.__class__.__name__}(d_model={self.d_model}, eps={self.eps})"


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_head: int,
        kv_heads: int,
        q_heads: int,
        seq_len: int,
        causal: bool = False,
        do_flash: bool = False,
    ) -> None:
        super().__init__()
        assert q_heads > kv_heads
        assert q_heads % kv_heads == 0

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d_head = d_head

        self.q_proj = nn.Linear(d_model, q_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary_emb = ROPE(d_head=d_head)

        self.causal = causal
        self.do_flash = do_flash
        if self.causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones((seq_len, seq_len)))[None, None, :, :]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, d_model = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, seq_len, self.q_heads, self.d_head)
        K = K.view(B, seq_len, self.kv_heads, self.d_head)
        V = V.view(B, seq_len, self.kv_heads, self.d_head)

        Q = self.rotary_emb(Q)
        K = self.rotary_emb(K)

        K = K.repeat_interleave(self.q_heads // self.kv_heads, dim=2)
        V = V.repeat_interleave(self.q_heads // self.kv_heads, dim=2)

        Q = Q.transpose(1, 2)  # (B, N_head, S, D_head)
        K = K.transpose(1, 2)  # (B, N_head, S, D_head)
        V = V.transpose(1, 2)  # (B, N_head, S, D_head)

        if self.do_flash:
            attn = torch.nn.functional.scaled_dot_product_attention(
                query=Q,
                key=K,
                value=V,
                attn_mask=(self.mask if self.causal else None),
                is_causal=self.causal,
            )
        else:
            attn_scores = (
                Q @ K.transpose(-1, -2) / self.d_head**0.5
            )  # (B, N_head, S, S)
            if self.causal:
                attn_scores.masked_fill(self.mask == 0, float("-inf"))
            attn = attn_scores.softmax(dim=-1) @ V  # (B, N_head, S, d_head)
            attn = attn.transpose(1, 2)  # (B, S, N_head, d_head)

        attn = attn.reshape(B, seq_len, d_model)
        return self.o_proj(attn)


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        kv_heads: int,
        q_heads: int,
        seq_len: int,
        causal: bool = False,
        do_flash: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            d_head=d_head,
            kv_heads=kv_heads,
            q_heads=q_heads,
            seq_len=seq_len,
            causal=causal,
            do_flash=do_flash,
        )
        self.mlp = LlamaMLP(d_model=d_model)
        self.input_layernorm = RMSNorm(d_model=d_model)
        self.post_attention_layernorm = RMSNorm(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_layernorm(x)
        attn = self.self_attn(x_norm)
        attn += x

        ffn_norm = self.post_attention_layernorm(attn)
        ffn_out = self.mlp(ffn_norm)
        ffn_out += attn

        return ffn_out


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.d_model
        )

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                d_model=config.d_model,
                d_head=config.d_head,
                kv_heads=config.kv_heads,
                q_heads=config.q_heads,
                seq_len=config.seq_len,
                causal=config.causal,
                do_flash=config.do_flash,
            )
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(d_model=config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embed_tokens(x)
        for layer in self.layers:
            x_emb = layer(x_emb)

        x_norm = self.norm(x_emb)

        return self.lm_head(x_norm)

    @classmethod
    def from_pretrained(cls, model_id: str) -> nn.Module:
        with open(f"configs/{model_id}.yaml", "r") as f:
            config = yaml.safe_load(f)

        llama_config = LlamaConfig(**config)
        model = cls(config=llama_config)
        hf_model = AutoModelForCausalLM.from_pretrained(model_id)

        # EMBEDDING TOKENS
        model.embed_tokens.weight = hf_model.model.embed_tokens.weight

        # DECODER LAYERS
        for torch_layer, hf_layer in zip(model.layers, hf_model.model.layers):
            # SELF ATTENTION BLOCK
            torch_layer.self_attn.q_proj.weight = hf_layer.self_attn.q_proj.weight
            torch_layer.self_attn.k_proj.weight = hf_layer.self_attn.k_proj.weight
            torch_layer.self_attn.v_proj.weight = hf_layer.self_attn.v_proj.weight
            torch_layer.self_attn.o_proj.weight = hf_layer.self_attn.o_proj.weight

            # MLP BLOCK
            torch_layer.mlp.gate_proj.weight = hf_layer.mlp.gate_proj.weight
            torch_layer.mlp.up_proj.weight = hf_layer.mlp.up_proj.weight
            torch_layer.mlp.down_proj.weight = hf_layer.mlp.down_proj.weight

            # NORMS
            torch_layer.input_layernorm.gain.weight = hf_layer.input_layernorm.weight
            torch_layer.post_attention_layernorm.gain.weight = (
                hf_layer.post_attention_layernorm.weight
            )

        # OUT NORM
        model.norm.gain.weight = hf_model.model.norm.weight  # type: ignore
        # LM HEAD
        model.lm_head.weight = hf_model.lm_head.weight

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B")

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    model = Llama.from_pretrained(model_id=args.model_id)
    print(model)
    # hf_model = AutoModelForCausalLM.from_pretrained(args.model_id)
    #
    # tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    #
    # prompt = "What is the capital of France ?"
    # input_ids = tokenizer.encode(prompt, return_tensors="pt")
    #
    # hf_logits = hf_model(input_ids)
    # last_hf_token = hf_logits.logits[:, -1, :]
    # hf_pref_token = torch.argmax(last_hf_token, dim=-1)
    #
    # print(hf_pref_token)
    #
    # logits = model(input_ids)
    # last_token = logits[:, -1, :]
    # pref_token = torch.argmax(last_token, dim=-1)
    #
    # print(pref_token)
