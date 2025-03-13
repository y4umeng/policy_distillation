import argparse
import os
import gymnasium as gym
import ale_py
from rl_zoo3.utils import get_model_path
from stable_baselines3 import DQN
from src.distillers._base import Distiller, Vanilla
from src.engine.utils import load_checkpoint
from src.engine.envpool_val import validate_async, preprocess_env
from src.models import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="dqn")
    parser.add_argument("-e", "--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument("-n", "--num_envs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for async EnvPool")
    parser.add_argument("-eps", "--episodes", type=int, default=100)
    args = parser.parse_args()

    # Register ALE envs so gymnasium recognizes them
    gym.register_envs(ale_py)

    # Create an EnvPool environment (e.g. in async mode)
    env = preprocess_env(
        args.env, 
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        async_mode=True,      # or False if you want synchronous
    )

    # Load your model
    model, path = get_model(args.model, args.env, env.action_space.n)
    if args.ckpt != "pretrain":
        path = args.ckpt
    model_state_dict = load_checkpoint(path)["model"]
    model.load_state_dict(model_state_dict)
    model.to("cuda")

    distiller = Vanilla(model)

    # Validate
    total_score = validate_async(
        distiller, 
        env, 
        args.num_envs,
        num_episodes=args.episodes, 
    )
    print(f"Total score over {args.episodes} episodes: {total_score}")
    print(f"Average score: {total_score / args.episodes}")
