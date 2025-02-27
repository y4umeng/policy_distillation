import argparse
import os
import gymnasium as gym
import ale_py
from rl_zoo3.utils import get_model_path
from stable_baselines3 import DQN
# from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from src.distillers._base import Distiller, Vanilla
from src.engine.utils import validate, preprocess_env, load_checkpoint
from src.models.breakout import breakout_model_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="dqn")
    parser.add_argument("-e", "--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument("-eps", "--episodes", type=int, default=100)
    args = parser.parse_args()


    if args.env == "BreakoutNoFrameskip-v4":
        model, path = breakout_model_dict[args.model]
        if args.ckpt != "pretrain":
            path = args.ckpt
        model_state_dict = load_checkpoint(path)["model"]
        model = model()
        model.load_state_dict(model_state_dict)
        model.to("cuda")
    else:
        raise NotImplementedError()
    
    distiller = Vanilla(model)
    
    gym.register_envs(ale_py)
    env = preprocess_env(args.env)

    total_score = validate(distiller, env, args.episodes)
    print(f"Total score over {args.episodes} episodes: {total_score}")