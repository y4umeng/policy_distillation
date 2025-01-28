import argparse
import yaml
import torch

from rl_distiller.distillers import distiller_dict
from rl_distiller.engine.trainer import RLTrainer
from rl_distiller.env.atari_wrappers import make_atari_env
from rl_distiller.models.custom_cnn import CustomCNN
from stable_baselines3 import PPO

def main(config_file):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    env_id = cfg["ENV"]["ID"]
    seed = cfg["ENV"]["SEED"]
    wrappers = cfg["ENV"]["WRAPPERS"]

    # Make environment
    env = make_atari_env(env_id, seed=seed, wrappers=wrappers)

    # Teacher: load from SB3 or your checkpoint
    teacher_name = cfg["DISTILLER"]["TEACHER"]
    # For demonstration, assume teacher is also a PPO model on disk
    teacher_model = PPO.load(f"./pretrained_teachers/{teacher_name}")  # example path

    # Student: either new or loaded
    student_model = PPO("CnnPolicy", env, verbose=1)
    # Optionally plug in our custom CNN
    # from stable_baselines3.common.torch_layers import NatureCNN
    # or custom architecture:
    # student_model.policy.features_extractor = CustomCNN(env.observation_space, features_dim=512)

    # Build distiller
    distiller_type = cfg["DISTILLER"]["TYPE"]
    DistillerClass = distiller_dict.get(distiller_type, distiller_dict["NONE"])
    distiller = DistillerClass(student=student_model, teacher=teacher_model, cfg=cfg)

    # trainer
    trainer = RLTrainer(distiller=distiller, env=env, cfg=cfg["SOLVER"])
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()
    main(args.cfg)
