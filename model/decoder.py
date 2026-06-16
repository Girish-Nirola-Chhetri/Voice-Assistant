import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.config import D_MODEL, NUM_HEADS


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.masked_attention = MultiHeadAttention(D_MODEL, NUM_HEADS)

        self.cross_attention = MultiHeadAttention(D_MODEL, NUM_HEADS)

        self.ffn = FeedForward()

        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.norm3 = nn.LayerNorm(D_MODEL)

    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None
    ):

        masked_output = self.masked_attention(
            x,
            mask=tgt_mask
        )

        x = self.norm1(x + masked_output)

        cross_output = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output
        )

        x = self.norm2(x + cross_output)

        ffn_output = self.ffn(x)

        x = self.norm3(x + ffn_output)

        return x