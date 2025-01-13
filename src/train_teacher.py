import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import logging

from config import Config
from teacher_network import DQN
from experience import ReplayBuffer
from gymnasium.wrappers import FrameStackObservation, ResizeObservation, GrayscaleObservation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def preprocess_env(env_id):
    """
    Preprocess the environment: resize, grayscale, and frame stack.
    """
    env = gym.make(env_id)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, 4)
    return env

def select_epsilon_greedy_action(net, state, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        with torch.no_grad():
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)
            q_vals = net(state_v)
            _, act_v = torch.max(q_vals, dim=1)
            return int(act_v.item())


def dqn_loss(policy_net, target_net, batch):
    states, actions, rewards, next_states, dones, _ = batch
    states_v = torch.tensor(states, dtype=torch.float32).to(Config.DEVICE)
    actions_v = torch.tensor(actions, dtype=torch.long).to(Config.DEVICE)
    rewards_v = torch.tensor(rewards, dtype=torch.float32).to(Config.DEVICE)
    next_states_v = torch.tensor(next_states, dtype=torch.float32).to(Config.DEVICE)
    dones_t = torch.tensor(dones, dtype=torch.bool).to(Config.DEVICE)

    q_values = policy_net(states_v)
    q_value = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_q_values = target_net(next_states_v)
        next_q_value = torch.max(next_q_values, dim=1)[0]
        next_q_value[dones_t] = 0.0  # zero for done transitions

    expected_q_value = rewards_v + Config.TEACHER_GAMMA * next_q_value
    return F.mse_loss(q_value, expected_q_value)


def train_teacher(env_id=Config.ENV_ID):
    logging.info("Initializing environment and networks...")
    env = preprocess_env(env_id)
    num_actions = env.action_space.n

    # Teacher & target networks
    teacher_net = DQN(in_channels=Config.FRAME_STACK, num_actions=num_actions).to(Config.DEVICE)
    target_net = DQN(in_channels=Config.FRAME_STACK, num_actions=num_actions).to(Config.DEVICE)
    target_net.load_state_dict(teacher_net.state_dict())

    optimizer = optim.Adam(teacher_net.parameters(), lr=Config.TEACHER_LR)
    replay_buffer = ReplayBuffer(Config.TEACHER_REPLAY_SIZE)

    epsilon = Config.TEACHER_START_EPSILON
    epsilon_decay = (Config.TEACHER_START_EPSILON - Config.TEACHER_END_EPSILON) / Config.TEACHER_EPSILON_DECAY_FRAMES

    state, _ = env.reset()
    total_frames = 0

    # For logging episode rewards
    episode_reward = 0.0
    episode_count = 0
    episode_rewards = []

    logging.info("Starting teacher training...")

    for frame_idx in range(Config.TEACHER_MAX_FRAMES):
        total_frames += 1
        epsilon = max(Config.TEACHER_END_EPSILON, epsilon - epsilon_decay)

        # Epsilon-greedy action
        action = select_epsilon_greedy_action(teacher_net, state, epsilon, num_actions)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store in replay buffer
        replay_buffer.push(state, action, reward, next_state, done, teacher_probs=None)

        state = next_state
        episode_reward += reward

        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)

            # Log info about the finished episode
            logging.info(
                f"Episode: {episode_count}, "
                f"Frame: {frame_idx}, "
                f"Epsilon: {epsilon:.4f}, "
                f"Episode Reward: {episode_reward:.2f}"
            )

            # Reset for the next episode
            episode_reward = 0.0
            state, _ = env.reset()

        # Training step
        if len(replay_buffer) > Config.TEACHER_BATCH_SIZE:
            batch = replay_buffer.sample(Config.TEACHER_BATCH_SIZE)
            loss = dqn_loss(teacher_net, target_net, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log training loss periodically
            if frame_idx % 1000 == 0:
                logging.info(f"Frame: {frame_idx}, Training Loss: {loss.item():.4f}")

        # Update target network
        if frame_idx % Config.TEACHER_TARGET_UPDATE == 0:
            target_net.load_state_dict(teacher_net.state_dict())

    # Save the teacher network
    torch.save(teacher_net.state_dict(), "teacher_dqn.pth")
    env.close()
    logging.info("Teacher training finished.")
    logging.info("Teacher network saved to teacher_dqn.pth")

    # Optionally log some statistics, e.g., average of the last 100 episode rewards
    if len(episode_rewards) > 0:
        avg_reward = np.mean(episode_rewards[-100:])
        logging.info(f"Average reward over the last 100 episodes: {avg_reward:.2f}")
    else:
        logging.info("No complete episodes were recorded.")
