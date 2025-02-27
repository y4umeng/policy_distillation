import os
import argparse
import copy
import torch.nn as nn
import torch.backends.cudnn as cudnn
from src.engine.cfg import show_cfg

import gymnasium as gym
import ale_py
from rl_zoo3.utils import get_model_path
from stable_baselines3 import DQN
from src.distillers import distiller_dict
from src.engine.utils import preprocess_env, load_checkpoint, create_experiment_name
from src.engine.cfg import CFG as cfg
from src.engine import trainer_dict
from src.models.breakout import breakout_model_dict

cudnn.benchmark = True

def main(cfg, resume, opts):
    experiment_name, tags = create_experiment_name(cfg, opts)
    if cfg.LOG.WANDB:
        try:
            import wandb
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print("Failed to use WANDB", "INFO")
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)

    # init env
    gym.register_envs(ale_py)
    env = preprocess_env(cfg.DISTILLER.ENV)

    # init models
    if cfg.DISTILLER.TYPE in distiller_dict:
        teacher, teacher_path = breakout_model_dict[cfg.DISTILLER.TEACHER]

        teacher_state_dict = load_checkpoint(teacher_path)["model"]
        teacher = teacher()
        teacher.load_state_dict(teacher_state_dict)
        teacher.to("cuda")

        student, _ = breakout_model_dict[cfg.DISTILLER.STUDENT]
        student = student().to("cuda")
        
        # student
        distiller = distiller_dict[cfg.DISTILLER.TYPE](student, teacher, cfg)
    else:
        raise NotImplementedError()

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, env, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)