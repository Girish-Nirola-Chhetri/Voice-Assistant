import json
import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm

from model.mask import create_causal_mask
from model.transformer import Transformer
from model.config import (
    LEARNING_RATE,
    EPOCHS,
    VOCAB_SIZE
)

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load("tokenizer/assistant_tokenizer.model")

with open("data/processed/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model = Transformer()

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for sample in data:

        source_text = sample["command"]

        target_text = json.dumps(
            sample["json"],
            separators=(",", ":")
        )

        source_ids = tokenizer.Encode(
            source_text,
            out_type=int
        )

        target_ids = [
            tokenizer.bos_id()
        ] + tokenizer.Encode(
            target_text,
            out_type=int
        ) + [
            tokenizer.eos_id()
        ]

        if len(target_ids) < 2:
            continue

        source = torch.tensor(
            [source_ids],
            dtype=torch.long
        )

        target = torch.tensor(
            [target_ids],
            dtype=torch.long
        )

        target_input = target[:, :-1]
        target_output = target[:, 1:]

        target_mask = create_causal_mask(
            target_input.size(1)
        )


        output = model(
            source,
            target_input,
            target_mask
        )

        predictions = torch.argmax(
            output,
            dim=-1
        )

        correct = (
            predictions == target_output
        ).sum().item()

        total_correct += correct
        total_tokens += target_output.numel()

        output = output.reshape(
            -1,
            VOCAB_SIZE
        )

        target_output = target_output.reshape(-1)

        loss = criterion(
            output,
            target_output
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data)

    accuracy = (
        total_correct /
        total_tokens
    ) * 100

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )

torch.save(
    model.state_dict(),
    "assistant_model.pt"
)

print("Training Complete!")