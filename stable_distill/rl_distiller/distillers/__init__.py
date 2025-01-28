from ._base import Distiller
from .kd import RL_KD

distiller_dict = {
    "NONE": Distiller,  # acts as a vanilla baseline (no KD)
    "KD": RL_KD,
}
