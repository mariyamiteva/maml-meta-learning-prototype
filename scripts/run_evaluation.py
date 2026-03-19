from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.maml import MAML
from src.model import SimpleNet
from src.tasks import sample_meta_batch
from src.utils import load_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAML evaluation on sampled tasks.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to JSON config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    set_seed(config["seed"])

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

    tasks = sample_meta_batch(
        meta_batch_size=train_cfg["num_eval_tasks"],
        n_way=task_cfg["n_way"],
        k_shot=task_cfg["k_shot"],
        q_query=task_cfg["q_query"],
        input_dim=task_cfg["input_dim"],
        device=device,
    )

    loss, acc = maml.evaluate(tasks)
    print(f"eval_loss={loss:.4f} eval_acc={acc:.4f}")


if __name__ == "__main__":
    main()
