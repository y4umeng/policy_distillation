# config.py
import torch

class Config:
    # ENV
    ENV_ID = "ALE/Breakout-v5"  # Updated for Gymnasium
    FRAME_STACK = 4
    FRAME_SIZE = (84, 84)

    # TEACHER DQN
    TEACHER_BATCH_SIZE = 32
    TEACHER_GAMMA = 0.99
    TEACHER_LR = 1e-4
    TEACHER_REPLAY_SIZE = 100_000
    TEACHER_START_EPSILON = 1.0
    TEACHER_END_EPSILON = 0.01
    TEACHER_EPSILON_DECAY_FRAMES = 1_000_000
    TEACHER_TARGET_UPDATE = 10_000
    TEACHER_MAX_FRAMES = 200_000  # Adjust as needed

    # STUDENT DISTILLATION
    STUDENT_BATCH_SIZE = 32
    STUDENT_LR = 1e-4
    STUDENT_EPOCHS = 3
    STUDENT_ALPHA = 1.0  # Temperature for teacher's softmax

    # MISC
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
