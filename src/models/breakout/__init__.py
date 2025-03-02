from .dqn import QNetwork, QNetwork1, QNetwork2, QNetwork3, QNetwork4
import os

model_path = "download_ckpts/BreakoutNoFrameskip-v4/"

breakout_model_dict = {
    "dqn": (QNetwork, os.path.join(model_path, "dqn")),
    "dqn1": (QNetwork1, None),
    "dqn2": (QNetwork2, None),
    "dqn3": (QNetwork3, os.path.join(model_path, "dqn3_pd")),
    "dqn4": (QNetwork4, os.path.join(model_path, "dqn4_pd"))
}