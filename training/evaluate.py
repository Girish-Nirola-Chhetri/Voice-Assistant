import torch
import sentencepiece as spm

from model.transformer import Transformer
from training.mask import create_causal_mask


MODEL_PATH = "assistant_model.pt"
TOKENIZER_PATH = "tokenizer/assistant_tokenizer.model"

MAX_LENGTH = 64


tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(TOKENIZER_PATH)

model = Transformer()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


def generate(command):

    source_ids = tokenizer.Encode(
        command,
        out_type=int
    )

    source = torch.tensor(
        [source_ids],
        dtype=torch.long
    )

    generated = [tokenizer.bos_id()]

    with torch.no_grad():

        for _ in range(MAX_LENGTH):

            target = torch.tensor(
                [generated],
                dtype=torch.long
            )

            target_mask = create_causal_mask(
                target.size(1)
            )

            output = model(
                source,
                target,
                target_mask
            )

            next_token = torch.argmax(
                output[:, -1, :],
                dim=-1
            ).item()

            generated.append(
                next_token
            )

            if next_token == tokenizer.eos_id():
                break

    generated = generated[1:]

    if generated and generated[-1] == tokenizer.eos_id():
        generated = generated[:-1]

    return tokenizer.Decode(generated)


while True:

    command = input("\nCommand: ")

    if command.lower() == "exit":
        break

    prediction = generate(command)

    print("\nPrediction:")
    print(prediction)