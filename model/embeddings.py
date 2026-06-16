import torch 
import torch.nn as nn

from config import VOCAB_SIZE, D_MODEL

class TokenEmbedding( nn.Module ) :
    def __init__(self, VOCAB_SIZE, D_MODEL ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings= VOCAB_SIZE,
            embedding_dim= D_MODEL
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        return self.embedding(x)



# example 

# embed = TokenEmbedding( VOCAB_SIZE=200, D_MODEL=128)

# tokens = torch.tensor([[1,34,56,23]])

# output = embed( tokens )
# print( output)
# print("\n", tokens)