import argparse
import os
import torch
import gymnasium as gym
import ale_py
from rl_zoo3.utils import get_model_path
from stable_baselines3 import DQN
# from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from src.distillers._base import Distiller, Vanilla
from src.engine.utils import validate, preprocess_env, load_checkpoint, save_checkpoint
from src.models import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="dqn")
    parser.add_argument("-e", "--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument("-eps", "--episodes", type=int, default=100)
    args = parser.parse_args()


    if args.model == "dqn":
        _, model_path, log_path = get_model_path(
            0,
            "rl-trained-agents",
            args.model,
            args.env
        )
        model = DQN.load(model_path).policy.q_net



        if args.ckpt != "pretrain":
            student_state = load_checkpoint(args.ckpt)["model"]
            model.load_state_dict(student_state)
    else:
        raise NotImplementedError()
    
    gym.register_envs(ale_py)
    env = preprocess_env(args.env)
    
    distiller = Vanilla(model)

    print(f"Total params: {distiller.get_learnable_parameters()}")
    print(distiller.student)
    model, _ = get_model(args.model, args.env, env.action_space.n)
    model.load_state_dict(distiller.student.state_dict())

    state = {"model": model.state_dict()}

    path = f"../download_ckpts/{args.env}/{args.model}"

    save_checkpoint(state, path)

    model.load_state_dict(load_checkpoint(path)["model"])
    
    distiller = Vanilla(model)
    distiller.to("cuda")

    print(distiller.student)
    

    total_score = validate(distiller, env, args.episodes)
    print(f"Total score over {args.episodes} episodes: {total_score}")