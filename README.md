# Policy Distillation for Atari Environments

This repository implements policy distillation techniques applied to Atari game environments. The framework demonstrates how to train a Deep Q-Network (DQN) teacher model and distill its policy into a smaller, more efficient student network using PyTorch and Gymnasium. This branch is modified to use envpool.

---

## Overview

Policy distillation is a method to transfer the knowledge of a large, pre-trained teacher model into a compact student model. This project offers a modular, configurable toolkit for:
- Training a DQN teacher on Atari environments.
- Distilling the teacher’s policy into a student network.
- Evaluating model performance using standardized benchmarks.

Key components include data generation via replay buffers, adjustable training schedules, and logging with optional Weights & Biases integration.

---

## Directory Structure

```plaintext
.
├── src
│   ├── configs
│   │   └── BreakoutNoFrameskip-v4
│   │       └── dqn.yaml         # Experiment configuration for Breakout (DQN)
│   ├── distillers
│   │   ├── __init__.py
│   │   ├── _base.py            # Base distillation class (defines training & inference)
│   │   └── PD.py               # (Optional) Additional policy distillation methods
│   ├── engine
│   │   ├── __init__.py
│   │   ├── cfg.py              # Global configuration and experiment setup
│   │   ├── experience.py       # Replay buffer for storing experiences
│   │   ├── trainer.py          # Training loop and epoch management
│   │   └── utils.py            # Utility functions (logging, LR scheduling, etc.)
│   ├── tools
│   │   ├── __init__.py
│   │   ├── eval.py             # Evaluation script for trained models
│   │   └── train.py            # Training script for distillation
│   └── load_teacher.py         # Utility to load a pre-trained teacher model
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Required Python packages and versions
└── setup.py                    # Package setup script
```

---

## Features

- **Teacher-Student Distillation**: Transfer knowledge from a large DQN teacher to a smaller student network.
- **Modular Design**: Clean separation between configuration, model distillation, training engine, and evaluation tools.
- **Atari Environments**: Built for experiments on Gymnasium Atari environments (e.g., Breakout).
- **Configurable Experiments**: Customize hyperparameters and training settings via YAML config files and command-line overrides.
- **Logging & Checkpointing**: Integrated logging (with optional Weights & Biases support) and robust checkpointing for resuming experiments.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/policy-distillation.git
cd policy-distillation
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Install Atari ROMs

For Atari environments, install the Arcade Learning Environment (ALE) ROMs using AutoROM:

```bash
pip install autorom
AutoROM --accept-license
```

### 4. Set Up the Package (Optional)

For development, install the package locally:

```bash
pip install -e .
```

---

## Usage

### Training the Distillation Model

Start the training (distillation) process by running the training script. You can specify a configuration file and override any settings via command-line options:

```bash
python src/tools/train.py --cfg src/configs/BreakoutNoFrameskip-v4/dqn.yaml --opts EXPERIMENT.NAME "my_experiment"
```

- Use the `--resume` flag to continue training from the latest checkpoint.
- Additional options can be passed to override default parameters in the config.

### Evaluating the Model

Evaluate a trained model using the evaluation script:

```bash
python src/tools/eval.py --model dqn --env BreakoutNoFrameskip-v4 --ckpt pretrain --episodes 10
```

This script loads the pre-trained teacher model, runs evaluation episodes, and prints the total score.

### Loading a Pretrained Teacher

The `load_teacher.py` script demonstrates how to load a pretrained teacher model (from a checkpoint or Hugging Face hub):

```bash
python src/load_teacher.py --env BreakoutNoFrameskip-v4 --algo dqn
```

---

## Configuration

- **YAML Configs**: Experiment-specific configurations are stored in `src/configs/BreakoutNoFrameskip-v4/dqn.yaml`. Modify this file to adjust settings like experiment tags, logging options, and hyperparameters.
- **Global Config**: Additional default parameters and configurations are defined in `src/engine/cfg.py`.

---

## Key Dependencies

- **PyTorch**: For building and training neural networks.
- **Gymnasium**: For creating and interacting with Atari environments.
- **Stable Baselines3 & rl_zoo3**: For pre-trained models and environment wrappers.
- **Weights & Biases (wandb)**: For experiment tracking (optional).

---

## Citation

If you find this project useful, please cite the original paper:

```bibtex
@inproceedings{policy_distillation,
  title={Policy Distillation},
  author={Rusu, Andrei A. and Gómez Colmenarejo, Sergio and Gulcehre, Caglar and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2016}
}
```

---

## Acknowledgments

This project is inspired by the [Policy Distillation](https://arxiv.org/abs/1511.06295) paper and builds upon the DQN architecture introduced in [Mnih et al. (2015)](https://www.nature.com/articles/nature14236).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute, report issues, or suggest improvements!
