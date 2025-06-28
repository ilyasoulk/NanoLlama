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
        self.down_proj = nn.Linear(hidden_dim, hidden_dim)
        self.swiglu = Swiglu(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self.up_proj(x)
        activations = self.swiglu(x_up)
        return self.down_proj(activations)


class ROPE(nn.Module):
    def __init__(self, d_model: int, base: float = 10_000.0) -> None:
        super().__init__()
        freq = 1.0 / (base) ** (torch.arange(0, d_model, 2) / d_model)  # (D / 2,)
        self.register_buffer("freq", freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(0) % 2 == 0  # (B, S, N_head, D_head)
        B, S, N_head, D_head = x.shape
        x_complex = x.view(B, S, N_head, D_head // 2, 2)

        pos = torch.arange(0, S, device=x.device, dtype=x.dtype)  # (S,)
        theta = torch.outer(pos, self.freq)  # (S, D/2)
        cos, sin = (
            theta.cos()[None, :, None, :],
            theta.sin()[None, :, None, :],
        )  # (1, S, 1, D/2)

        x1, x2 = x.unbind(-1)
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

        return out


class RMSNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GroupedQueryAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
