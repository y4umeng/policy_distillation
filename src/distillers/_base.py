import torch
import torch.nn as nn
import torch.nn.functional as F

class Distiller(nn.Module):
    def __init__(self, student, teacher, cfg):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.cfg = cfg
    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student = self.student(image)
        target /= self.cfg.SOLVER.SOFTMAX_TEMP
        target = torch.softmax(target, dim=1)
        loss = F.mse_loss(logits_student, target)
        return logits_student, {"mse": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)