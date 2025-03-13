import envpool
import torch
import numpy as np
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing
from tqdm import tqdm
import os

def validate(distiller, env, num_episodes=10, bar=True):
    total_reward = 0
    if bar: pbar = tqdm(range(num_episodes))
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        print(f'STATE: {state.min()}')
        while not done:
            # Convert the state to a tensor and get the action
            state_v = torch.tensor(state, dtype=torch.float32, device="cuda").squeeze().unsqueeze(0)
            with torch.no_grad():
                q_values = distiller.forward_test(state_v)
                action = q_values.argmax(dim=1).item()  # Select action with max Q-value

            # Step in the environment
            print(f"ACTION: {type(action)}")
            next_state, reward, terminated, truncated, _ = env.step(np.array(action))
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
        if bar:
            pbar.set_description(f"Total score: {total_reward}")
            pbar.update()
    if bar: pbar.close()
    return total_reward

def preprocess_env(env_name, num_envs=1, task_id="default"):
    """
    Preprocess the environment: resize, grayscale, and frame stack, then parallelize it using envpool.
    """
    # Use envpool.make with proper arguments, including task_id
    print("USING ENVPOOL")
    if "NoFrameskip" in env_name:
        env_name = env_name.replace("NoFrameskip", "").replace("4", "5")
        print(env_name)
    env = envpool.make(
        env_type="gymnasium",    # Envpool's gym-based environments
        task_id=env_name,   # task_id is now required
        num_envs=num_envs, # Number of parallel environments
        # frame_skip=1
    )
    # Apply Gym wrappers for Atari preprocessing
    # env = AtariPreprocessing(env)
    # env = FrameStackObservation(env, 4)  # Stack the last 4 frames for observation

    return env
