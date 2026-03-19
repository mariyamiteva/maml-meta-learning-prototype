from dataclasses import dataclass


@dataclass
class TaskConfig:
    n_way: int
    k_shot: int
    q_query: int
    input_dim: int
