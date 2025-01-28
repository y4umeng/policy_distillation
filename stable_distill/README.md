# RL-Distiller

This repository demonstrates a knowledge distillation framework for **reinforcement learning** using [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). It is inspired by the MDistiller repository structure.

## Features

- **Distillers** for RL: Compare student policy distributions against a pretrained teacher’s distributions (PPO, DQN, etc.).
- **Config-based** approach with example YAMLs in `configs/`.
- **Trainer** that integrates stable-baselines3 calls with distillation logic.

## Setup

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
