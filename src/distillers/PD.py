import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

class PD(Distiller):
    def forward_train(self, image, target, **kwargs):
        target /= self.cfg.SOLVER.SOFTMAX_TEMP
        target = torch.softmax(target, dim=1)
        logits_student = self.student(image)
        logits_student /= self.cfg.SOLVER.SOFTMAX_TEMP
        logits_student = torch.softmax(logits_student, dim=1)
        loss = loss = F.kl_div(logits_student.log(), target, reduction='batchmean')
        return logits_student, {"kld": loss}