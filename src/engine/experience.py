# experience.py
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset, DataLoader

class ReplayBufferDataset(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, teacher_probs):
        """
        Add a new action to the buffer.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, teacher_probs.squeeze()))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        # Return the transition at the given index.
        return self.buffer[index]