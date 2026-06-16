import math
import torch
import torch.nn as nn
from config import NUM_HEADS, D_MODEL

class MultiHeadAttention( nn.Module ) :
    def __init__( self, D_MODEL, NUM_HEADS ):
        super().__init__()

        assert D_MODEL % NUM_HEADS == 0

        self.d_model = D_MODEL
        self.num_heads = NUM_HEADS
        self.head_dim = D_MODEL // NUM_HEADS

        self.query = nn.Linear(D_MODEL, D_MODEL)
        self.key = nn.Linear( D_MODEL, D_MODEL)
        self.value = nn.Linear( D_MODEL, D_MODEL)

        self.fc_output = nn.Linear( D_MODEL, D_MODEL)               # fully connected output

    
    def forward(self, query, key=None, value=None, mask=None):
        if key is None:
            key = query

        if value is None:
            value = query
            
        batch_size = query.size(0)

        query_len = query.size(1)
        key_len = key.size(1)

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)


        Q = Q.reshape( batch_size, query_len, self.num_heads, self.head_dim).transpose(1,2)       # final => batch_size, num_heads, seq_len , head_dim
        K = K.reshape( batch_size, key_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.reshape( batch_size, key_len, self.num_heads, self.head_dim).transpose(1,2)

        scores = Q @ K.transpose(-1, -2)

        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        output = weights @ V
        output = output.transpose(1,2)

        # concatenate the head
        output = output.contiguous().reshape(
            batch_size,
            query_len,
            self.d_model
        )

        output = self.fc_output( output )

        return output
