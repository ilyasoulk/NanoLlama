import math
from typing import Tuple, Union
import yaml
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import custom_op
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LlamaConfig:
    d_model: int
    vocab_size: int
    intermediate_size: int
    kv_heads: int
    q_heads: int
    d_head: int
    num_layers: int
    seq_len: int
    rms_norm_eps: float
    batch_size: int
    rope_theta: float
    causal: bool = False
    do_flash: bool = False


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        y = x.double()
        return (y * F.sigmoid(y)).to(input_dtype)


@custom_op("myops::silu", mutates_args=())
def fused_silu(x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, d_model: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.act_fn = SiLU()  # This doesn't work with the HF implem, might need to check torch._C._nn.silu...
        # ongoing investigation, on cuda the drift is not as large, only breaks at 20 layers deep
        # on fp64 it goes deeper
        # self.act_fn.compile()
        # https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/functional.py#L2358
        # self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x_up)


class ROPE(nn.Module):
    def __init__(self, d_head: int, base: float = 10_000.0) -> None:
        super().__init__()
        freq = 1.0 / (
            (base)
            ** (
                torch.arange(0, d_head, 2, dtype=torch.int64).to(dtype=torch.float)
                / d_head
            )
        )  # (D / 2,)
        self.register_buffer("inv_freq", freq)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @torch.no_grad()
    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S = q.size(2)
        pos = torch.arange(0, S, device=q.device, dtype=torch.float32)  # (S,)
        theta = torch.outer(pos, self.inv_freq)  # (S, D / 2)
        theta = torch.cat((theta, theta), dim=-1)
        cos, sin = (
            theta.cos()[None, None, :, :],
            theta.sin()[None, None, :, :],
        )  # (1, 1, S, D/2)

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)

        return q_rot, k_rot


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.gain * x).to(input_dtype)

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
        rope_theta: float,
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

        self.rotary_emb = ROPE(d_head=d_head, base=rope_theta)

        self.causal = causal
        self.do_flash = do_flash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, d_model = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, seq_len, self.q_heads, self.d_head)
        K = K.view(B, seq_len, self.kv_heads, self.d_head)
        V = V.view(B, seq_len, self.kv_heads, self.d_head)

        Q = Q.transpose(1, 2)  # (B, N_head, S, D_head)
        K = K.transpose(1, 2)  # (B, N_head, S, D_head)
        V = V.transpose(1, 2)  # (B, N_head, S, D_head)

        Q_rot, K_rot = self.rotary_emb(Q, K)

        # This could be removed and setting enable_gqa in sdpa
        K_rot = K_rot.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        V = V.repeat_interleave(self.q_heads // self.kv_heads, dim=1)

        if self.do_flash:
            attn = torch.nn.functional.scaled_dot_product_attention(
                query=Q_rot,
                key=K_rot,
                value=V,
                is_causal=self.causal,
            )
        else:
            # Set do_flash to true for now
            # Investigate why my own attention doesn't pass the tests,
            # maybe take a look at
            # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            attn_bias = torch.zeros(
                (seq_len, seq_len), device=Q_rot.device, dtype=Q_rot.dtype
            )
            if self.causal:
                mask = torch.tril(
                    torch.ones(
                        (seq_len, seq_len), device=Q_rot.device, dtype=torch.bool
                    ),
                    diagonal=0,
                )
                attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
                attn_bias.to(Q_rot.dtype)

            scale = 1 / math.sqrt(self.d_head)
            attn_scores = Q_rot @ K_rot.transpose(-1, -2) * scale  # (B, N_head, S, S)
            attn_scores += attn_bias
            attn = attn_scores.softmax(dim=-1) @ V  # (B, N_head, S, d_head)

        attn = attn.transpose(1, 2).contiguous()  # (B, S, N_head, d_head)
        attn = attn.reshape(B, seq_len, d_model)
        return self.o_proj(attn)


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        intermediate_size: int,
        kv_heads: int,
        q_heads: int,
        seq_len: int,
        rms_norm_eps: float,
        rope_theta: float,
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
            rope_theta=rope_theta,
            causal=causal,
            do_flash=do_flash,
        )
        self.mlp = LlamaMLP(d_model=d_model, intermediate_size=intermediate_size)
        self.input_layernorm = RMSNorm(d_model=d_model, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(d_model=d_model, eps=rms_norm_eps)

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
                intermediate_size=config.intermediate_size,
                kv_heads=config.kv_heads,
                q_heads=config.q_heads,
                seq_len=config.seq_len,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                causal=config.causal,
                do_flash=config.do_flash,
            )
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(d_model=config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self, x: torch.Tensor, output_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        hidden_states = list()
        x_emb = self.embed_tokens(x)
        for layer in self.layers:
            if output_hidden_states:
                hidden_states.append(x_emb)
            x_emb = layer(x_emb)

        x_norm = self.norm(x_emb)

        if output_hidden_states:
            return self.lm_head(x_norm), hidden_states

        return self.lm_head(x_norm)

    @classmethod
    def from_pretrained(cls, model_id: str) -> nn.Module:
        with open(f"configs/{model_id}.yaml", "r") as f:
            config = yaml.safe_load(f)

        llama_config = LlamaConfig(**config)
        model = cls(config=llama_config)
        model = model.to(torch.float32)
        hf_model = AutoModelForCausalLM.from_pretrained(model_id)
        hf_model = hf_model.to(torch.float32)

        # EMBEDDING TOKENS
        model.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()

        # DECODER LAYERS
        for torch_layer, hf_layer in zip(model.layers, hf_model.model.layers):
            # SELF ATTENTION BLOCK
            torch_layer.self_attn.q_proj.weight.data = (
                hf_layer.self_attn.q_proj.weight.data.clone()
            )
            torch_layer.self_attn.k_proj.weight.data = (
                hf_layer.self_attn.k_proj.weight.data.clone()
            )
            torch_layer.self_attn.v_proj.weight.data = (
                hf_layer.self_attn.v_proj.weight.data.clone()
            )
            torch_layer.self_attn.o_proj.weight.data = (
                hf_layer.self_attn.o_proj.weight.data.clone()
            )

            # MLP BLOCK
            torch_layer.mlp.gate_proj.weight.data = (
                hf_layer.mlp.gate_proj.weight.data.clone()
            )
            torch_layer.mlp.up_proj.weight.data = (
                hf_layer.mlp.up_proj.weight.data.clone()
            )
            torch_layer.mlp.down_proj.weight.data = (
                hf_layer.mlp.down_proj.weight.data.clone()
            )

            # NORMS
            torch_layer.input_layernorm.gain.data = (
                hf_layer.input_layernorm.weight.data.clone()
            )
            torch_layer.post_attention_layernorm.gain.data = (
                hf_layer.post_attention_layernorm.weight.data.clone()
            )

        # OUT NORM
        model.norm.gain.data = hf_model.model.norm.weight.data.clone()  # type: ignore
        # LM HEAD
        model.lm_head.weight.data = hf_model.lm_head.weight.data.clone()

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    model = Llama.from_pretrained(model_id=args.model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    layer_idx = 0

    prompt = "What is the capital of france ?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    logits, hidden_states = model(input_ids, output_hidden_states=True)
    outputs = hf_model(input_ids, output_hidden_states=True, return_dict=True)

    hf_hidden_states = outputs.hidden_states

    for hf_layer, torch_layer in zip(hf_hidden_states, hidden_states):
        print(f"Layer : {layer_idx}")
        torch.testing.assert_close(torch_layer, hf_layer, rtol=1e-4, atol=1e-5)
        layer_idx += 1

    torch.testing.assert_close(logits, outputs.logits)
