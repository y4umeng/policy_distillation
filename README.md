# Policy Distillation for Atari Environments

This repository implements the *Policy Distillation* approach described in the paper [Policy Distillation](https://arxiv.org/abs/1511.06295) by Rusu et al. (2016). The project includes training a Deep Q-Network (DQN) teacher model on Atari games and distilling its policy into a smaller student network for improved efficiency and multi-task capabilities.

---

## Project Structure

```plaintext
.
├── config.py               # Configuration file for hyperparameters and environment settings
├── train_teacher.py        # Script for training the DQN teacher model
├── distill_student.py      # Script for distilling the teacher model's policy into a student model
├── test_teacher.py         # Script for testing the trained teacher model
├── teacher_network.py      # Implementation of the DQN teacher network
├── student_network.py      # Implementation of the student network
├── experience.py           # Replay buffer for storing experience tuples
├── README.md               # Project documentation (this file)
└── requirements.txt        # List of required Python libraries
```

---

## Features

- **Teacher Model Training**: Train a DQN model using Gymnasium Atari environments.
- **Policy Distillation**: Distill the teacher's policy into a smaller student network for improved efficiency.
- **Multi-Task Learning**: Combine policies from multiple tasks into a single network (optional extension).
- **Evaluation**: Test the trained models on their respective environments.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/policy-distillation.git
cd policy-distillation
```

### 2. Install Dependencies

Use the provided `requirements.txt` file to install the necessary dependencies.

```bash
pip install -r requirements.txt
```

### 3. Install Atari ROMs

Gymnasium requires the Arcade Learning Environment (ALE) ROMs for Atari games. Install them using `AutoROM`:

```bash
pip install autorom
AutoROM --accept-license
```

---

## Usage

### 1. Train the Teacher Model

Train a DQN teacher model on a Gymnasium Atari environment.

```bash
python train_teacher.py
```

By default, the teacher model is saved as `teacher_dqn.pth`.

### 2. Distill the Student Model

Distill the trained teacher's policy into a smaller student network.

```bash
python distill_student.py
```

By default, the student model is saved as `student_policy.pth`.

### 3. Test the Trained Teacher

Evaluate the performance of the trained teacher model on the Atari environment.

```bash
python test_teacher.py --model-path teacher_dqn.pth --env-name ALE/Breakout-v5 --episodes 5
```

---

## Configuration

Modify `config.py` to customize:
- Environment settings (e.g., `ENV_ID`).
- Hyperparameters (e.g., learning rate, batch size).
- Frame preprocessing options (e.g., stacking, resizing).

---

## Key Dependencies

- **Gymnasium**: For creating and interacting with Atari environments.
- **PyTorch**: For building and training neural networks.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this project, please cite the original paper:

```plaintext
@inproceedings{policy_distillation,
  title={Policy Distillation},
  author={Andrei A. Rusu, Sergio Gómez Colmenarejo, Caglar Gulcehre, et al.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2016}
}
```

---

## Acknowledgments

This project is based on the [Policy Distillation](https://arxiv.org/abs/1511.06295) paper by Google DeepMind and follows the DQN architecture described in [Mnih et al. (2015)](https://www.nature.com/articles/nature14236).

---
