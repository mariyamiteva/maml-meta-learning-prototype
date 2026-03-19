from __future__ import annotations

from typing import Any

from src.maml import MAML
from src.model import SimpleNet
from src.tasks import sample_meta_batch


def train(config: dict[str, Any]) -> None:
    device = config["device"]
    task_cfg = config["task"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    model = SimpleNet(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["output_dim"],
    ).to(device)

    maml = MAML(
        model=model,
        inner_lr=train_cfg["inner_lr"],
        meta_lr=train_cfg["meta_lr"],
        inner_steps=train_cfg["inner_steps"],
    )

    for step in range(1, train_cfg["meta_steps"] + 1):
        train_tasks = sample_meta_batch(
            meta_batch_size=task_cfg["meta_batch_size"],
            n_way=task_cfg["n_way"],
            k_shot=task_cfg["k_shot"],
            q_query=task_cfg["q_query"],
            input_dim=task_cfg["input_dim"],
            device=device,
        )

        meta_loss, meta_acc = maml.meta_step(train_tasks)

        if step == 1 or step % train_cfg["eval_every"] == 0:
            eval_tasks = sample_meta_batch(
                meta_batch_size=train_cfg["num_eval_tasks"],
                n_way=task_cfg["n_way"],
                k_shot=task_cfg["k_shot"],
                q_query=task_cfg["q_query"],
                input_dim=task_cfg["input_dim"],
                device=device,
            )
            eval_loss, eval_acc = maml.evaluate(eval_tasks)
            print(
                f"step={step:04d} "
                f"train_meta_loss={meta_loss:.4f} "
                f"train_meta_acc={meta_acc:.4f} "
                f"eval_loss={eval_loss:.4f} "
                f"eval_acc={eval_acc:.4f}"
            )
        else:
            print(
                f"step={step:04d} "
                f"train_meta_loss={meta_loss:.4f} "
                f"train_meta_acc={meta_acc:.4f}"
            )
