import torch
import torch.nn.functional as F
from ._base import Distiller

class PD(Distiller):
    def kld_with_temp(self, student_logits, teacher_logits, temp):
        teacher_logits /= temp
        teacher_logits = torch.softmax(teacher_logits, dim=1)
        student_logits /= temp
        student_logits = torch.softmax(student_logits, dim=1)
        return F.kl_div(student_logits.log(), teacher_logits, reduction='batchmean')
    def forward_train(self, image, target, **kwargs):
        logits_student = self.student(image)
        loss = self.kld_with_temp(logits_student, target, self.cfg.SOLVER.SOFTMAX_TEMP)
        return logits_student, {"kld": loss}