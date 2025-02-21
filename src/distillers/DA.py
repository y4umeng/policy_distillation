import torch
import torch.nn as nn
import torch.nn.functional as F
from .PD import PD

class DA(PD):
    def DA(self, images):
        images.requires_grad_(True)
        optimizer = torch.optim.Adam([images], lr=self.cfg.DA.LR)
        for _ in range(self.cfg.DA.EPOCHS):
            logits_student = self.student(images)
            logits_teacher = self.teacher(images)
            loss = -1 * self.kld_with_temp(
                        logits_student, logits_teacher
                    )
            # images.grad = torch.autograd.grad(loss, images)[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return images.detach()

    def forward_train(self, image, target, **kwargs):
        if torch.rand(1)[0] < self.cfg.DA.PROB:
            image = self.DA(image)
            target = self.teacher(image)
        logits_student = self.student(image)
        loss = self.kld_with_temp(logits_student, target)
        return logits_student, {"kld": loss}