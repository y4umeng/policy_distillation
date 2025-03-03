from .dqn import QNetwork, QNetwork1, QNetwork2, QNetwork3, QNetwork4
import os

model_path = "download_ckpts/BreakoutNoFrameskip-v4/"

model_dict = {
    "dqn": QNetwork,
    "dqn1": QNetwork1,
    "dqn2": QNetwork2,
    "dqn3": QNetwork3,
    "dqn4": QNetwork4
}

breakout_checkpoint_dict = {
    "dqn": os.path.join(model_path, "dqn"),
    "dqn3": os.path.join(model_path, "dqn3_pd"),
    "dqn4": os.path.join(model_path, "dqn4_pd")
}

checkpoint_dicts = {
    "BreakoutNoFrameskip-v4": breakout_checkpoint_dict
}

def get_model(model_name, env_name, num_actions):
    model = model_dict[model_name]
    model = model(num_actions=num_actions)
    if env_name in checkpoint_dicts and model_name in checkpoint_dicts[env_name]:
        path = checkpoint_dicts[env_name][model_name]
        return model, path
    else:
        return model, None

