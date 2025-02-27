from .dqn import QNetwork, QNetworkSmall

breakout_model_dict = {
    "dqn": (QNetwork, "download_ckpts/BreakoutNoFrameskip-v4/dqn"),
    "dqn_small": (QNetworkSmall, None)
}