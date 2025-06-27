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


class Rope(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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
