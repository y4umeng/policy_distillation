from ._base import Distiller
from .PD import PD

distiller_dict = {
    "NONE": Distiller,
    "PD": PD,
}