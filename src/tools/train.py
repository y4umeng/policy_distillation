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
from src.distillers._base import Distiller
from src.engine.utils import reinitialize_model_weights, preprocess_env
from src.engine.cfg import CFG as cfg
from src.engine import trainer_dict

cudnn.benchmark = True

def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
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
    if cfg.DISTILLER.TYPE == "NONE":

        # load teacher
        _, model_path, _ = get_model_path(
            0,
            "rl-trained-agents",
            cfg.DISTILLER.TEACHER,
            cfg.DISTILLER.ENV
        )
         
        if cfg.DISTILLER.TEACHER == "dqn":
            teacher = DQN.load(model_path).policy.q_net
        else:
            raise NotImplementedError()
        
        # student
        if cfg.DISTILLER.STUDENT == "dqn":
            student = reinitialize_model_weights(teacher)
            distiller = Distiller(student)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, teacher, distiller, env, cfg
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