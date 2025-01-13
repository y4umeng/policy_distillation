# experience.py
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, teacher_probs=None):
        """
        'done' here is a single boolean that unifies
        (terminated or truncated) from Gymnasium.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, teacher_probs)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, teacher_probs = map(np.array, zip(*batch))
        return state, action, reward, next_state, done, teacher_probs

    def __len__(self):
        return len(self.buffer)
