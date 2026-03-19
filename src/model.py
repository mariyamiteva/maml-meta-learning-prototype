from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        params: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

        x = F.linear(x, params["fc1.weight"], params["fc1.bias"])
        x = F.relu(x)
        x = F.linear(x, params["fc2.weight"], params["fc2.bias"])
        x = F.relu(x)
        x = F.linear(x, params["fc3.weight"], params["fc3.bias"])
        return x

    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {name: param for name, param in self.named_parameters()}
