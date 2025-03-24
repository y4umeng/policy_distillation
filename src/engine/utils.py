import gymnasium as gym
import torch
import numpy as np
import time
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing
from tqdm import tqdm
import os

def validate(distiller, env, num_episodes=10, bar=True, wandb_name=""):
    if wandb_name:
        import wandb
        wandb.init(project="policy_distillation_evals", name=wandb_name)

    episode_rewards = []
    if bar:
        pbar = tqdm(range(num_episodes))
        
    for episode in range(num_episodes):
        start_time = time.time()
        
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Convert the state to a tensor and get the action
            state_v = torch.tensor(state, dtype=torch.float32, device="cuda").squeeze().unsqueeze(0)
            with torch.no_grad():
                q_values = distiller.forward_test(state_v)
                action = q_values.argmax(dim=1).item()  # Select action with max Q-value

            # Step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)
        
        # Log per-episode reward and runtime if using wandb
        if wandb_name:
            episode_runtime = time.time() - start_time
            # Log average reward and standard deviation across all episodes
            avg_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            wandb.log({
                "episode_reward": episode_reward,
                "episode_runtime": episode_runtime,
                "avg_reward": avg_reward,
                "std_reward": std_reward
            }, step=episode)

        if bar:
            pbar.set_description(f"Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f}")
            pbar.update()

    if bar:
        pbar.close()

    return sum(episode_rewards)


def preprocess_env(env_name, num_envs=1):
    """
    Preprocess the environment: resize, grayscale, and frame stack.
    """
    env = gym.make(env_name)
    env = AtariPreprocessing(env)
    env = FrameStackObservation(env, 4)
    return env

def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def create_experiment_name(cfg, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    return experiment_name, tags

def save_checkpoint(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)