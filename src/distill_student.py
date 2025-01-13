import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from config import Config
from teacher_network import DQN
from student_network import StudentNet1
from experience import ReplayBuffer
from train_teacher import preprocess_env

def generate_distillation_data(env_id=Config.ENV_ID, num_samples=50_000):
    """
    Roll out the teacher's policy to collect states and 
    teacher's action distribution (softmax over Q-values).
    """
    env = preprocess_env(env_id)
    num_actions = env.action_space.n

    # Load the teacher
    teacher_net = DQN(in_channels=Config.FRAME_STACK, num_actions=num_actions).to(Config.DEVICE)
    teacher_net.load_state_dict(torch.load("teacher_dqn.pth", map_location=Config.DEVICE))
    teacher_net.eval()

    distill_buffer = ReplayBuffer(capacity=num_samples)

    state, _ = env.reset()
    collected = 0

    while collected < num_samples:
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            q_vals = teacher_net(state_v)  # shape [1, num_actions]
            policy_dist = F.softmax(q_vals / Config.STUDENT_ALPHA, dim=1).cpu().numpy()[0]

        teacher_action = np.argmax(policy_dist)

        next_state, reward, terminated, truncated, _ = env.step(teacher_action)
        done = terminated or truncated

        # Store in buffer
        distill_buffer.push(
            state,
            teacher_action,
            reward,
            next_state,
            done,
            teacher_probs=policy_dist
        )

        state = next_state
        collected += 1

        if done:
            state, _ = env.reset()

    env.close()
    return distill_buffer


def distill_student(env_id=Config.ENV_ID):
    distill_buffer = generate_distillation_data(env_id=env_id, num_samples=50_000)
    env = preprocess_env(env_id)
    num_actions = env.action_space.n
    env.close()

    student_net = StudentNet1(in_channels=Config.FRAME_STACK, num_actions=num_actions).to(Config.DEVICE)
    optimizer = optim.Adam(student_net.parameters(), lr=Config.STUDENT_LR)

    for epoch in range(Config.STUDENT_EPOCHS):
        total_loss = 0.0
        steps = 0

        indices = np.arange(len(distill_buffer))
        np.random.shuffle(indices)

        batch_size = Config.STUDENT_BATCH_SIZE
        for start_idx in range(0, len(indices), batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            batch = [distill_buffer.buffer[i] for i in batch_indices]

            states, _, _, _, _, teacher_probs = zip(*batch)
            states_v = torch.tensor(states, dtype=torch.float32).to(Config.DEVICE)
            teacher_probs_v = torch.tensor(teacher_probs, dtype=torch.float32).to(Config.DEVICE)

            logits = student_net(states_v)
            log_probs = F.log_softmax(logits, dim=1)

            loss = -(teacher_probs_v * log_probs).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"Epoch {epoch+1}/{Config.STUDENT_EPOCHS} | Loss: {total_loss/steps:.4f}")

    torch.save(student_net.state_dict(), "student_policy.pth")
    print("Distillation complete. Student model saved as student_policy.pth")
