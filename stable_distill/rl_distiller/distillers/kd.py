import torch
import torch.nn.functional as F

from ._base import Distiller

class RL_KD(Distiller):
    """
    Simple demonstration of distilling from a teacher policy distribution.
    For example, if teacher is a stable-baselines3 PPO,
    we can compare action logits or value function outputs.
    """

    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.alpha = cfg.KD.ALPHA
        self.temperature = cfg.KD.TEMPERATURE

    def distill_step(self, obs, student_action, teacher_action, **kwargs):
        """
        Called after we have both student's and teacher's chosen actions
        or action distributions for a given state/obs.
        """
        # Example: teacher logits vs. student logits:
        # teacher_logits = teacher_action.logits  # placeholder
        # student_logits = student_action.logits

        # For stable-baselines3, we can do something like:
        #   teacher_policy = self.teacher.policy.forward(obs)  # -> distribution
        #   student_policy = self.student.policy.forward(obs)  # -> distribution
        #   teacher_logits = teacher_policy.distribution.logits
        #   student_logits = student_policy.distribution.logits

        # Example cross-entropy based KD:
        # Use temperature scaling
        # teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        # log_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        # kd_loss = F.kl_div(log_student, teacher_probs, reduction="batchmean") * (self.temperature ** 2)

        # Weighted by alpha
        # return self.alpha * kd_loss

        # For demonstration, return 0.0:
        kd_loss = 0.0
        return kd_loss
