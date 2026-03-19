# MAML Prototype for Few-Shot Learning

A lightweight, repo-ready implementation of **Model-Agnostic Meta-Learning (MAML)** for few-shot classification in PyTorch.

This project is designed as a clean, reproducible prototype that demonstrates the core mechanics of meta-learning:
- training on a **distribution of tasks**
- performing **task-specific inner-loop adaptation**
- optimizing shared initialization parameters for **fast adaptation to unseen tasks**

It is intentionally compact and readable, so it can serve both as a research companion repository and as a base for further experimentation.

## Methodology

### Problem setup
The implementation uses an **N-way K-shot** few-shot classification formulation.

Each meta-learning task is sampled independently and contains:
- a **support set** used for adaptation
- a **query set** used for meta-optimization

For each task:
- `N` = number of classes per task
- `K` = number of support examples per class
- `Q` = number of query examples per class

The project includes a synthetic task generator based on Gaussian class clusters so the full pipeline can be executed without external datasets.

### MAML training procedure

#### 1. Shared initialization
The model starts with shared parameters \(\theta\), learned across many tasks.

#### 2. Inner loop adaptation
For a sampled task, the model performs one or more gradient steps on the support set:

$$
\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{support}(\theta)
$$

where:
- $\alpha$ is the inner-loop learning rate  
- $\theta'$ are the adapted task-specific parameters

#### 3. Outer loop meta-update
The adapted parameters are then evaluated on the query set:

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{query}(\theta'_i)
$$

Gradients from the query loss are backpropagated through the adaptation process to update the original shared initialization \(\theta\).

### Design choices
- **Second-order MAML**: the implementation keeps the inner loop differentiable via `create_graph=True`
- **Synthetic tasks**: ensures deterministic, dependency-light experimentation
- **Minimal MLP backbone**: keeps the focus on meta-learning rather than model complexity
- **Functional forward path**: supports adaptation with task-specific parameter dictionaries

### What this repository demonstrates
- task-based training instead of sample-based training
- rapid adaptation to new tasks from a learned initialization
- clean separation between:
  - task generation
  - model definition
  - meta-learning algorithm
  - training loop
  - evaluation

## Repository structure

```text
maml-prototype/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── tasks.py
│   ├── model.py
│   ├── maml.py
│   ├── train.py
│   └── utils.py
├── configs/
│   └── default.json
├── scripts/
│   ├── run_training.py
│   └── run_evaluation.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run training

```bash
python scripts/run_training.py --config configs/default.json
```

## Run evaluation

```bash
python scripts/run_evaluation.py --config configs/default.json
```

## Example output

Training logs show meta-loss and query accuracy trends over time. Evaluation reports mean accuracy across held-out meta-test tasks after fast adaptation.

## Configuration
All major hyperparameters are controlled from `configs/default.json`, including:
- task definition (`n_way`, `k_shot`, `q_query`)
- model dimensions
- inner-loop and outer-loop learning rates
- number of meta-training steps
- number of inner adaptation steps

## Extensions
This prototype is intentionally compact, but it can be extended with:
- first-order MAML
- convolutional backbones
- real few-shot datasets such as Omniglot or miniImageNet
- meta-validation and early stopping
- experiment tracking
- adaptation visualizations

## Notes
This repository is a methodological prototype. It is not intended as a benchmark implementation, but as a readable and runnable reference for the MAML training paradigm.
