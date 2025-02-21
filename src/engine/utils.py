import gymnasium as gym
import torch
import copy
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gymnasium.wrappers import FrameStackObservation
from tqdm import tqdm

def validate(distiller, env, num_episodes=100, bar=True):
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

def preprocess_env(env_name):
    """
    Preprocess the environment: resize, grayscale, and frame stack.
    """
    env = gym.make(env_name)
    env = AtariWrapper(env)
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

def reinitialize_model_weights(model):
    """
    Copies and reinitializes the weights of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to copy and reinitialize.

    Returns:
        torch.nn.Module: A copy of the model with reinitialized weights.
    """
    copied_model = copy.deepcopy(model)
    for module in copied_model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    return copied_model

def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)