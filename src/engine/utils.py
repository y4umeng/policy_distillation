import gymnasium as gym
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
        total_reward += episode_reward
        if bar:
            pbar.set_description(log_msg(f"Total score: {total_reward}", "EVAL"))
            pbar.update()
    if bar: pbar.close()
    return total_reward

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
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)