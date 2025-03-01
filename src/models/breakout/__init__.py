from .dqn import QNetwork, QNetwork1, QNetwork2, QNetwork3, QNetwork4

breakout_model_dict = {
    "dqn": (QNetwork, "download_ckpts/BreakoutNoFrameskip-v4/dqn"),
    "dqn1": (QNetwork1, None),
    "dqn2": (QNetwork2, None),
    "dqn3": (QNetwork3, None),
    "dqn4": (QNetwork4, None)
}