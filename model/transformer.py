import torch
import torch.nn as nn
from config import D_MODEL, NUM_LAYERS, VOCAB_SIZE

from embeddings import TokenEmbedding
from positional_encoding import PositionalEncoder
from encoder import Encoder
from decoder import DecoderBlock

class Transformer( nn.Module ):
    def __init__(self):
        
        self.embedding = TokenEmbedding()

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

