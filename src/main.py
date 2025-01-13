# main.py
import gymnasium as gym
import ale_py
from train_teacher import train_teacher
from distill_student import distill_student

if __name__ == "__main__":

    gym.register_envs(ale_py)

    # 1) Train the teacher DQN
    train_teacher()  # Will save "teacher_dqn.pth"

    # 2) Distill the student policy
    distill_student()  # Will save "student_policy.pth"

    print("Done!")
