# Codigo baseado no codigo 
# https://github.com/KristinaRay/Deep-Learning-School-part-2/blob/main/modules.py
#
# Data: 09/05/2024
#
# Modificado por: 
# - Luan Matheus Trindade Dalmazo 

# =============================================================================
#  Header
# =============================================================================

import torch 
import torch.nn as nn

# =============================================================================
#  Encoder
# =============================================================================

''' creating our model where:
[input_dim] is the size/dimensionality of the one-hot vectors t
hat will be input to the encoder. This is equal to the input (source) vocabulary size.
[embedding_dim] is the dimensionality of the embedding layer. This layer converts 
the one-hot vectors into dense vectors with embedding_dim dimensions.
[hidden_dim] is the dimensionality of the hidden and cell states.
[n_layers] is the number of layers in the RNN.
[dropout] is the amount of dropout to use. This is a regularization parameter to prevent overfitting.'''
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell