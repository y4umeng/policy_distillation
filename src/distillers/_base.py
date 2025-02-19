import torch.nn as nn
import torch.nn.functional as F

class Distiller(nn.Module):
    def __init__(self, student):
        super(Distiller, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student = self.student(image)
        loss = nn.KLDivLoss(logits_student, target)
        return logits_student, {"kld": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)