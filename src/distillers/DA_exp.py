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
        torch.autograd.set_detect_anomaly(True)
        fake_student = copy.deepcopy(self.student)
        images /= 255 # do in place operation before turning grad on
        images.requires_grad = True
        self.teacher.train()
        optimizer = torch.optim.Adam([images], lr=self.cfg.DA.LR)
        for _ in range(self.cfg.DA.EPOCHS):
            logits_student = self.normalize(fake_student(images, normalize=False))
            logits_teacher = self.normalize(self.teacher(images, normalize=False))
            loss = -1 * self.kld_with_temp(
                        logits_student, logits_teacher, 1.0
                    )
            # loss = -1 * F.mse_loss(logits_student, logits_teacher)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.teacher.eval()
        images.requires_grad = False
        images = torch.clamp(images, min=0.0, max=1.0)
        return images.detach().clone(), loss

    def forward_train(self, image, target, **kwargs):
        if torch.rand(1)[0] < self.cfg.DA.PROB:
            image, _ = self.DA(image)
            print(f"IMAGE: {image.mean(), image.max(), image.min()}", flush=True)
            target = self.teacher(image, normalize=False)
            logits_student = self.student(image, normalize=False)
        else:
            logits_student = self.student(image)

        logits_student = self.normalize(logits_student)
        target = self.normalize(target)
        print(f"Loss: {logits_student.mean(), logits_student.min(), logits_student.max(), target.mean(), target.min(), target.max()}", flush=True)
        loss = self.kld_with_temp(logits_student, target, self.cfg.SOLVER.SOFTMAX_TEMP)
        print(f"LOSS: {loss}")
        if torch.isnan(loss):
            print("NAN LOSS", flush=True)
        return logits_student, {"kld": loss}