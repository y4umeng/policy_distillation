from ._base import Distiller, Vanilla
from .PD import PD
from .DA import DA
from .DA_MSE import DA_MSE

distiller_dict = {
    "NONE": Distiller,
    "PD": PD,
    "DA": DA,
    "DA_MSE": DA_MSE
}