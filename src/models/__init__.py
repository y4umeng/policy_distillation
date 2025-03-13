from .dqn import QNetwork, QNetwork1, QNetwork2, QNetwork3, QNetwork4, QNetwork5, QNetwork6
import os

model_dir = "download_ckpts/"

model_dict = {
    "dqn": QNetwork,
    "dqn1": QNetwork1,
    "dqn2": QNetwork2,
    "dqn3": QNetwork3,
    "dqn4": QNetwork4,
    "dqn5": QNetwork5,
    "dqn6": QNetwork6
}

breakout_checkpoint_dict = {
    "dqn3": os.path.join(model_dir, "BreakoutNoFrameskip-v4/dqn3_pd"),
    "dqn4": os.path.join(model_dir, "BreakoutNoFrameskip-v4/dqn4_pd")
}

checkpoint_dicts = {
    "BreakoutNoFrameskip-v4": breakout_checkpoint_dict
}

def get_model(model_name, env_name, num_actions, distiller=""):
    model = model_dict[model_name]
    model = model(num_actions=num_actions)
    if env_name in checkpoint_dicts and model_name in checkpoint_dicts[env_name]:
        path = checkpoint_dicts[env_name][model_name]
        return model, path
    else:
        path = os.path.join(model_dir, env_name, model_name)
        if distiller: path = path + "_" + distiller
        if os.path.exists(path):
            return model, path
        return model, None

