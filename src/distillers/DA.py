import torch
import torch.nn as nn
import torch.nn.functional as F
from .PD import PD
import copy


class DA(PD):    
    def DA(self, images):
        fake_student = copy.deepcopy(self.student)
        images.requires_grad = True
        optimizer = torch.optim.Adam([images], lr=self.cfg.DA.LR)
        for _ in range(self.cfg.DA.EPOCHS):
            logits_student = fake_student(images)
            logits_teacher = self.teacher(images)
            loss = -1 * self.kld_with_temp(
                        logits_student, logits_teacher, 1.0
                    )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return images.detach().clone(), loss

    def forward_train(self, image, target, **kwargs):
        if torch.rand(1)[0] < self.cfg.DA.PROB:
            image, _ = self.DA(image)
            target = self.teacher(image)
        logits_student = self.student(image)
        loss = self.kld_with_temp(logits_student, target, self.cfg.SOLVER.SOFTMAX_TEMP)
        if torch.isnan(loss):
            print("NAN LOSS", flush=True)
            raise ValueError()
        return logits_student, {"kld": loss}