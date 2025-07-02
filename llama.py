from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LlamaConfig:
    d_model: int
    vocab_size: int
    kv_heads: int
    q_heads: int
    d_head: int
    num_layers: int
    seq_len: int
    causal: bool = False


class Swiglu(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2 * self.d_model
        a = x[..., : self.d_model]
        b = x[..., self.d_model :]
        return a * (b * torch.sigmoid(b))


class MLP(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.swiglu = Swiglu(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.up_proj(x)
        activations = self.swiglu(x_up)
        return self.out_proj(activations)


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
        # import ipdb
        #
        # ipdb.set_trace()
        x1_pos = x1 * cos - x2 * sin
        x2_pos = x1 * sin + x2 * cos
        # out = torch.stack(
        #     (x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
        # )  # (B, S, N_head, D_head // 2, 2)
        out = torch.stack((x1_pos, x2_pos), dim=-1)

        return out.flatten(-2)  # (B, S, N_head, D_head)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x**2, keepdim=True, dim=-1).sqrt()
        return (x / (rms + self.eps)) * self.gain


class SelfAttention(nn.Module):
    def __init__(self, seq_len: int, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal
        if self.causal:
            self.register_buffer(
                "mask", torch.tril(torch.ones((seq_len, seq_len)))[None, None, :, :]
            )

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        # (B, S, N_head, D_head)
        assert len(Q.shape) == 4
        assert len(K.shape) == 4
        assert len(V.shape) == 4

        d_head = Q.size(-1)

        Q = Q.transpose(1, 2)  # (B, N_head, S, D_head)
        K = K.transpose(1, 2)  # (B, N_head, S, D_head)
        V = V.transpose(1, 2)  # (B, N_head, S, D_head)

        attn_scores = Q @ K.transpose(-1, -2) / d_head**0.5  # (B, N_head, S, S)

        if self.causal:
            attn_scores.masked_fill(self.mask == 0, float("-inf"))

        attn_scores = attn_scores.softmax(dim=-1) @ V  # (B, N_head, S, d_head)

        attn_scores = attn_scores.transpose(1, 2)  # (B, S, N_head, d_head)

        return attn_scores


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_head: int,
        kv_heads: int,
        q_heads: int,
        seq_len: int,
        causal: bool = False,
    ) -> None:
        super().__init__()
        assert q_heads > kv_heads
        assert q_heads % kv_heads == 0

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d_head = d_head

        self.W_q = nn.Linear(d_model, q_heads * d_head)
        self.W_kv = nn.Linear(d_model, 2 * kv_heads * d_head)

        self.rope = ROPE(d_head=d_head)
        self.self_attn = SelfAttention(seq_len=seq_len, causal=causal)

        self.W_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, seq_len, d_model = x.shape

        Q = self.W_q(x)
        kv = self.W_kv(x)
        K, V = kv.chunk(2, dim=-1)

        Q = Q.reshape(B, seq_len, self.q_heads, self.d_head)
        K = K.reshape(B, seq_len, self.kv_heads, self.d_head)
        V = V.reshape(B, seq_len, self.kv_heads, self.d_head)

        Q = self.rope(Q)
        K = self.rope(K)

        K = K.repeat_interleave(self.q_heads // self.kv_heads, dim=2)
        V = V.repeat_interleave(self.q_heads // self.kv_heads, dim=2)

        attn = self.self_attn(Q, K, V)

        attn = attn.reshape(B, seq_len, d_model)

        return self.W_out(attn)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        kv_heads: int,
        q_heads: int,
        seq_len: int,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model=d_model)
        self.ffn_norm = RMSNorm(d_model=d_model)
        self.GQA = GroupedQueryAttention(
            d_model=d_model,
            d_head=d_head,
            kv_heads=kv_heads,
            q_heads=q_heads,
            seq_len=seq_len,
            causal=causal,
        )
        self.ffn = MLP(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.attn_norm(x)
        attn = self.GQA(x_norm)
        attn += x

        ffn_norm = self.ffn_norm(attn)
        ffn_out = self.ffn(ffn_norm)
        ffn_out += attn

        return ffn_out


class LLama(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.d_model
        )

        self.seq = nn.Sequential(*[
            Encoder(
                d_model=config.d_model,
                d_head=config.d_head,
                kv_heads=config.kv_heads,
                q_heads=config.q_heads,
                seq_len=config.seq_len,
                causal=config.causal,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model=d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embeddings.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embeddings(x)
        x_out = self.seq(x_emb)

        x_norm = self.norm(x_out)

        return self.out(x_norm)


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )

    vocab_size = 10_000
    batch_size = 32
    seq_len = 128
    # -----
    d_model = 256
    num_layers = 2
    d_head = 32
    kv_heads = 4
    q_heads = 8

    config = LlamaConfig(
        num_layers=num_layers,
        d_model=d_model,
        vocab_size=vocab_size,
        d_head=d_head,
        kv_heads=kv_heads,
        q_heads=q_heads,
        seq_len=seq_len,
    )
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = LLama(config=config)

    model = model.to(device)

    token_ids = token_ids.to(device)

    logits = model(token_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size)
