import envpool
import torch
import numpy as np
from tqdm import tqdm

def validate_async(distiller, env, num_envs, num_episodes=10, bar=True):
    """
    Validate a Q-network (distiller) on an asynchronous envpool Atari environment.
    We run until `num_episodes` have completed in total across all parallel envs.
    """
    # The code assumes `env` was created in async mode:
    #   env = envpool.make(
    #       task_id="Pong-v5",
    #       env_type="gym", 
    #       num_envs=NUM_ENVS, 
    #       batch_size=BATCH_SIZE,
    #       ...
    #   )
    # e.g. env.async_reset()
    #
    # We'll track how many episodes are done in total. For each environment in
    # this async batch, we keep a reward accumulator. 
    
    # If the user wants to handle bar progress
    if bar:
        pbar = tqdm(total=num_episodes)
    
    completed_episodes = 0
    total_reward = 0.0
    
    # Each environment in EnvPool has an ID, 0..(num_envs-1).
    # We accumulate rewards for each env ID in this array:
    rewards_per_env = np.zeros(num_envs, dtype=np.float32)
    
    # Start everything by calling async_reset once
    env.async_reset()
    
    while completed_episodes < num_episodes:
        # Receive a batch of observations, rewards, terminations, truncations
        obs, rew, terminated, truncated, info = env.recv()
        env_id = info["env_id"]   # shape = (batch_size,)
        
        # Add the reward from this step
        rewards_per_env[env_id] += rew
        
        # Check which environments finished an episode
        done = terminated | truncated
        
        if np.any(done):
            done_indices = np.where(done)[0]
            for idx in done_indices:
                # The environment that just finished:
                i = env_id[idx]
                
                # Record that environment's total reward
                episode_reward = rewards_per_env[i]
                total_reward += episode_reward
                completed_episodes += 1
                
                # Optionally show progress
                if bar:
                    pbar.update()
                    pbar.set_description(f"Episode {completed_episodes} Reward={episode_reward:.2f}")
                
                # Reset that environment's reward count for its next new episode
                rewards_per_env[i] = 0.0
                
                # If we have completed enough episodes, stop
                if completed_episodes >= num_episodes:
                    break
        
        # Compute actions (a batch of actions) with your Q-network
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device="cuda")
            q_vals = distiller.forward_test(obs_t)
            # Argmax over actions
            actions = q_vals.argmax(dim=1).cpu().numpy()
        
        # Send actions back to exactly those envs that produced this batch
        env.send(actions, env_id)
    
    if bar:
        pbar.close()
    
    return total_reward

def preprocess_env(env_name, num_envs=1, batch_size=1, async_mode=True):
    """
    Example: create an EnvPool Atari environment in asynchronous mode. 
    Adjust `batch_size` and `num_envs` for your use case.
    """
    print("USING ENVPOOL (Async)" if async_mode else "USING ENVPOOL (Sync)")
    
    # Convert "PongNoFrameskip-v4" to "Pong-v5", etc., if needed:
    if "NoFrameskip" in env_name:
        env_name = env_name.replace("NoFrameskip", "").replace("4", "5")
    
    # Create an async environment using EnvPool
    if async_mode:
        env = envpool.make(
            env_type="gymnasium",
            task_id=env_name,
            num_envs=num_envs,  
            batch_size=batch_size,
            # For example, set these:
            # episodic_life=True,
            # repeat_action_probability=0.0,
            # etc. 
        )
    else:
        # Synchronous version if needed:
        env = envpool.make(
            env_type="gymnasium",
            task_id=env_name,
            num_envs=num_envs,
        )
    
    return env
