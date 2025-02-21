from ._base import Distiller
from .PD import PD
from .DA import DA

distiller_dict = {
    "NONE": Distiller,
    "PD": PD,
    "DA": DA
}