import torch

from model.embeddings import TokenEmbedding
from model.positional_encoding import PositionalEncoding


embed = TokenEmbedding(
    VOCAB_SIZE=200,
    D_MODEL=128
)

pos_encoder = PositionalEncoding(
    d_model=128
)

tokens = torch.tensor(
    [[1, 34, 56, 23]],
    dtype=torch.long
)
x = embed(tokens)

print("Embedding shape:", x.shape)

print("\nBefore positional encoding:")
print(x[0, 0, :10])

x_pos = pos_encoder(x)

print("\nAfter positional encoding:")
print(x_pos[0, 0, :10])

print("\nAfter positional encoding shape:", x_pos.shape)