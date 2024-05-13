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
#  Class Decoder
# =============================================================================

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(hidden_dim *2 + embedding_dim, hidden_dim, n_layers)

        self.energy = nn.Linear(hidden_dim *3, 1)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, input, encoder_states, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        
        rnn_input = torch.cat((context_vector, embedded), dim=2)
        # rnn_input: (1, N, hidden_size*2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell
