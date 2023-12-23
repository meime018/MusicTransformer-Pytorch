import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR, TransformerDecoderRPR, TransformerDecoderLayerRPR


# MusicTransformer
class MusicTransformer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Music Transformer reproduction from https://arxiv.org/abs/1809.04281. Arguments allow for
    tweaking the transformer architecture (https://arxiv.org/abs/1706.03762) and the rpr argument
    toggles Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class with dummy decoder to
    make a decoder-only transformer architecture

    For RPR support, there is modified Pytorch 1.2.0 code in rpr.py. Modified source will be
    kept up to date with Pytorch revisions only as necessary.
    ----------
    """

    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence=2048, rpr=False):
        super(MusicTransformer, self).__init__()

        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq    = max_sequence
        self.rpr        = rpr

        # Input embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, self.d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq)

        encoder_norm = LayerNorm(self.d_model)
        encoder1_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
        self.encoder1 = TransformerEncoderRPR(encoder1_layer, self.nlayers, encoder_norm)

        encoder2_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq)
        self.encoder2 = TransformerEncoderRPR(encoder2_layer, self.nlayers, encoder_norm)

        self.decoder_layer = TransformerDecoderLayerRPR(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.d_ff, dropout=self.dropout)
        self.decoder = TransformerDecoderRPR(self.decoder_layer, num_layers=self.nlayers, norm=encoder_norm)


        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, VOCAB_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

    # forward
    def forward(self, x1, x2, tgt, mask=True):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Takes an input sequence and outputs predictions using a sequence to sequence method.

        A prediction at one index is the "next" prediction given all information seen previously.
        ----------
        """

        if(mask is True):
            mask = generate_square_subsequent_mask(tgt.shape[1]).to(get_device())
        else:
            mask = None

        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        tgt = self.embedding(tgt)
        # Input shape is (max_seq, batch_size, d_model)
        x1 = x1.permute(1,0,2)
        x2 = x2.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        x1 = self.positional_encoding(x1)
        x2 = self.positional_encoding(x2)
        tgt = self.positional_encoding(tgt)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        memory1 = self.encoder1(src=x1)
        memory2 = self.encoder2(src=x2)
        x_out = self.decoder(tgt=tgt, memory1=memory1, memory2=memory2, tgt_mask=mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1,0,2)

        y = self.Wout(x_out)
        # y = self.softmax(y)

        del mask

        # They are trained to predict the next note in sequence (we don't need the last one)
        return y

    # generate


def generate_square_subsequent_mask(sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)