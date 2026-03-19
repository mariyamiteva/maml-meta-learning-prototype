from __future__ import annotations

import torch


def _generate_class_samples(mean: torch.Tensor, total_samples: int, std: float = 1.0) -> torch.Tensor:
    return mean.unsqueeze(0) + std * torch.randn(total_samples, mean.shape[0])


def create_synthetic_task(
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    input_dim: int = 20,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_samples = k_shot + q_query

    support_x, support_y = [], []
    query_x, query_y = [], []

    for class_idx in range(n_way):
        class_mean = torch.randn(input_dim, device=device) * 2.0
        class_samples = _generate_class_samples(class_mean, total_samples).to(device)

        support_x.append(class_samples[:k_shot])
        query_x.append(class_samples[k_shot:])
        support_y.append(torch.full((k_shot,), class_idx, dtype=torch.long, device=device))
        query_y.append(torch.full((q_query,), class_idx, dtype=torch.long, device=device))

    return (
        torch.cat(support_x, dim=0),
        torch.cat(support_y, dim=0),
        torch.cat(query_x, dim=0),
        torch.cat(query_y, dim=0),
    )


def sample_meta_batch(
    meta_batch_size: int,
    n_way: int,
    k_shot: int,
    q_query: int,
    input_dim: int,
    device: str = "cpu",
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        create_synthetic_task(
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            input_dim=input_dim,
            device=device,
        )
        for _ in range(meta_batch_size)
    ]
