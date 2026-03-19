from __future__ import annotations

import torch
import torch.nn.functional as F


class MAML:
    def __init__(
        self,
        model: torch.nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 1,
    ) -> None:
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)

    def clone_params(self) -> dict[str, torch.Tensor]:
        return {name: param for name, param in self.model.get_parameters().items()}

    def inner_update(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        adapted = params
        for _ in range(self.inner_steps):
            logits = self.model(support_x, adapted)
            loss = F.cross_entropy(logits, support_y)

            grads = torch.autograd.grad(
                loss,
                tuple(adapted.values()),
                create_graph=True,
            )

            adapted = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(adapted.items(), grads)
            }
        return adapted

    def meta_step(
        self,
        tasks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[float, float]:
        meta_loss = 0.0
        meta_accuracy = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            params = self.clone_params()
            adapted_params = self.inner_update(support_x, support_y, params)

            query_logits = self.model(query_x, adapted_params)
            query_loss = F.cross_entropy(query_logits, query_y)
            meta_loss = meta_loss + query_loss

            predictions = torch.argmax(query_logits, dim=1)
            accuracy = (predictions == query_y).float().mean()
            meta_accuracy = meta_accuracy + accuracy

        meta_loss = meta_loss / len(tasks)
        meta_accuracy = meta_accuracy / len(tasks)

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return float(meta_loss.item()), float(meta_accuracy.item())

    @torch.no_grad()
    def evaluate(
        self,
        tasks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[float, float]:
        losses = []
        accuracies = []

        for support_x, support_y, query_x, query_y in tasks:
            params = self.clone_params()
            adapted_params = self.inner_update_eval(support_x, support_y, params)

            query_logits = self.model(query_x, adapted_params)
            query_loss = F.cross_entropy(query_logits, query_y)

            predictions = torch.argmax(query_logits, dim=1)
            accuracy = (predictions == query_y).float().mean()

            losses.append(float(query_loss.item()))
            accuracies.append(float(accuracy.item()))

        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

    def inner_update_eval(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        adapted = {name: param.detach().clone().requires_grad_(True) for name, param in params.items()}
        for _ in range(self.inner_steps):
            logits = self.model(support_x, adapted)
            loss = F.cross_entropy(logits, support_y)
            grads = torch.autograd.grad(loss, tuple(adapted.values()), create_graph=False)
            adapted = {
                name: (param - self.inner_lr * grad).detach().clone().requires_grad_(True)
                for (name, param), grad in zip(adapted.items(), grads)
            }
        return adapted
