import torch
import torch.nn as nn


class Swiglu(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2 * self.hidden_dim
        a = x[..., : self.hidden_dim]
        b = x[..., self.hidden_dim :]
        return a * (b * torch.sigmoid(b))


class MLP(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.swiglu = Swiglu(hidden_dim=hidden_dim)

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
            theta.cos()[None, :, None, None, :],
            theta.sin()[None, :, None, None, :],
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x**2, keepdim=True, dim=-1).sqrt()
        return (x / rms + self.eps) * self.gain


class SelfAttention(nn.Module):
    def __init__(self, causal: bool = False) -> None:
        self.causal = causal
        super().__init__()

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
            # TODO : very inefficient mask, change this later
            mask = torch.ones_like(attn_scores) * -float("Inf")
            mask = torch.triu(mask)
            attn_scores += mask

        return attn_scores.softmax(-1) @ V


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, d_head: int, kv_heads: int, q_heads: int) -> None:
        super().__init__()
        assert d_model / kv_heads == d_head
        assert d_model / q_heads == d_head
        assert q_heads > kv_heads
        assert q_heads % kv_heads == 0

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d_head = d_head

        self.W_q = nn.Linear(d_model, q_heads * d_head)
        self.W_kv = nn.Linear(d_model, 2 * kv_heads * d_head)

        self.rope = ROPE(d_head=d_head)
        self.self_attn = SelfAttention()

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

        K = K.repeat_interleave(self.q_heads / self.kv_heads, dim=2)

        attn = self.self_attn(Q, K, V)

        attn = attn.reshape(B, seq_len, d_model)

        return self.W_out(attn)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LLama(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
