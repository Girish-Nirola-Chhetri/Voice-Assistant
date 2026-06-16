import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor()
tokenizer.load("tokenizer/assistant_tokenizer.model")

VOCAB_SIZE = tokenizer.get_piece_size()

D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 512
MAX_SEQ_LEN = 64

BATCH_SIZE = 8
LEARNING_RATE = 0.0003
EPOCHS = 100