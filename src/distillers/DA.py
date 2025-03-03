import torch
import torch.nn as nn
import torch.nn.functional as F
from .PD import PD
import copy


class DA(PD):
    def normalize(self, logit):
        mean = logit.mean(dim=-1, keepdims=True)
        stdv = logit.std(dim=-1, keepdims=True)
        return (logit - mean) / (1e-7 + stdv)
    
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
            # loss = -1 * F.mse_loss(logits_student, logits_teacher)
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