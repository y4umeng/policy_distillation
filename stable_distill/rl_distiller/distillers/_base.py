class Distiller:
    """
    Minimal base class for RL distillers.
    """

    def __init__(self, student, teacher=None):
        self.student = student
        self.teacher = teacher

    def train_mode(self):
        """Set student (and teacher, if any) to training mode."""
        self.student.train()
        if self.teacher:
            self.teacher.policy.set_training_mode(False)  # typically teacher is 'eval' mode in KD

    def get_student(self):
        return self.student

    def distill_step(self, obs, student_action, teacher_action, **kwargs):
        """
        Compute knowledge-distillation-related loss terms, if any.
        Typically you'd use the teacher's policy distribution
        to guide the student's updates.
        """
        return 0.0  # default no-op
