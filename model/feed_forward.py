import torch
import torch.nn as nn
from model.config import D_MODEL, FF_DIM

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FF_DIM),
            nn.ReLU(),
            nn.Linear( FF_DIM, D_MODEL)
        )

    def forward(self, x):
        return self.ffn( x )