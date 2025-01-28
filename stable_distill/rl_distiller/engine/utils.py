import numpy as np
import gym

def evaluate(model, env, episodes=5):
    """
    Evaluate the RL agent for the specified number of episodes and return average reward.
    """
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            ep_return += reward
        returns.append(ep_return)
    avg_return = np.mean(returns)
    return avg_return
