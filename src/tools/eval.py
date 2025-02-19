import argparse
import os
import gymnasium as gym
import ale_py
from rl_zoo3.utils import get_model_path
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from src.distillers._base import Distiller
from src.engine.utils import validate, preprocess_env, load_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="dqn")
    parser.add_argument("-e", "--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument("-eps", "--episodes", type=int, default=100)
    args = parser.parse_args()

    if args.ckpt == "pretrain":
        _, model_path, log_path = get_model_path(
            0,
            "rl-trained-agents",
            args.model,
            args.env
        )
        model = DQN.load(model_path).policy.q_net
        distiller = Distiller(model)
    else:
        # if args.model == "dqn":
        #     state = load_checkpoint(args.ckpt)
        #     distiller = Distiller(student=QNetwork())
        #     distiller.load_state_dict(state["model"])
        # else:
        raise NotImplementedError()
    
    gym.register_envs(ale_py)
    env = preprocess_env(args.env)

    total_score = validate(distiller, env, args.episodes)
    print(f"Total score over {args.episodes} episodes: {total_score}")