import torch
import torch.nn as nn
from model.config import D_MODEL, NUM_LAYERS, VOCAB_SIZE

from model.embeddings import TokenEmbedding
from model.positional_encoding import PositionalEncoder
from model.encoder import Encoder
from model.decoder import DecoderBlock

class Transformer( nn.Module ):
    def __init__(self):
        super().__init__()

        self.embedding = TokenEmbedding(D_MODEL, VOCAB_SIZE)

        self.positional_encoding = PositionalEncoder()

        self.encoder_layers = nn.ModuleList(
            Encoder()
            for _ in range ( NUM_LAYERS )
        )

        self.decoder_layers = nn.ModuleList(
            DecoderBlock()
            for _ in range ( NUM_LAYERS )
        )

        self.fc_output = nn.Linear(D_MODEL, VOCAB_SIZE )


    def encode( self, source): 

        x = self.embedding( source )
        x = self.positional_encoding( x )

        for layer in self.encoder_layers:
            x = layer( x )

        return x
    
    def decode(self, target, encoder_output, target_mask = None):

        x = self.embedding( target )
        x = self.positional_encoding( x )

        for layer in self.decoder_layers: 
            x = layer( x, encoder_output, target_mask )

        return x

    def forward( self, source, target, target_mask= None ):
        
        encoder_output = self.encode( source )

        decoder_output = self.decode( target, encoder_output, target_mask)


        output = self.fc_output( decoder_output )

        return output

