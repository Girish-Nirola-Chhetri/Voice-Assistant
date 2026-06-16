import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.config import D_MODEL, NUM_HEADS


class Encoder( nn.Module ):
    def __init__(self):
        super().__init__()

        self.attention = MultiHeadAttention(D_MODEL, NUM_HEADS)
        self.ffn = FeedForward()

        self.norm1 = nn.LayerNorm( D_MODEL )
        self.norm2 = nn.LayerNorm( D_MODEL )

    def forward(self, x):
        attention_output = self.attention( x )

        x = self.norm1( x + attention_output )

        ffn_output = self.ffn( x )

        x = self.norm2( x + ffn_output )

        return x