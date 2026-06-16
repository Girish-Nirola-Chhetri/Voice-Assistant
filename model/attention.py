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

    
    def forward( self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        Q = self.query( x )
        K = self.key( x )
        V = self.value( x )

        Q = Q.reshape( batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)       # final => batch_size, num_heads, seq_len , head_dim
        K = K.reshape( batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.reshape( batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        scores = Q @ K.transpose(-1, -2)

        # scaling
        scores = scores / math.sqrt(K.size(-1))

        weights = torch.softmax(
            scores,
            dim = -1
        )

        output = weights @ V
        output = output.transpose(1,2)

        # concatenate the head
        output = output.contiguous().reshape(
            batch_size, seq_len, self.d_model
        )

        output = self.fc_output( output )

        return output
