import argparse
import yaml
import torch
from rl_distiller.env.atari_wrappers import make_atari_env
from rl_distiller.engine.utils import evaluate
from stable_baselines3 import PPO

def main(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    env_id = cfg["ENV"]["ID"]
    seed = cfg["ENV"]["SEED"]
    wrappers = cfg["ENV"]["WRAPPERS"]

    env = make_atari_env(env_id, seed, wrappers)
    # load student model checkpoint
    checkpoint = cfg["SOLVER"]["SAVE_PATH"]
    model = PPO.load(checkpoint)
    
    avg_return = evaluate(model, env, episodes=cfg["SOLVER"]["EVAL_EPISODES"])
    print(f"Evaluation -> Average Return over {cfg['SOLVER']['EVAL_EPISODES']} episodes: {avg_return}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()
    main(args.cfg)
