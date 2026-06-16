import torch


def create_causal_mask(seq_len):
    mask = torch.tril(
        torch.ones(seq_len, seq_len)
    )

    return mask.unsqueeze(0).unsqueeze(0)