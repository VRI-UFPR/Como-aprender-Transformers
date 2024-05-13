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
import torch.optim as optim
import numpy as np
import utils
import spacy
import datasets
import torchtext
import tqdm
import evaluate
from dataset import SpeechDataset
from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq
import model_utils
import re 

# =============================================================================
#  Main
# =============================================================================

'''loading the dataset'''
valid_set = SpeechDataset('../seq2robot-dataset/dev.csv')
train_set = SpeechDataset('../seq2robot-dataset/train.csv')
test_set = SpeechDataset('../seq2robot-dataset/test.csv')

'''loading tokanizer'''
en_nlp = spacy.load("en_core_web_sm")

'''translating infos'''
max_length = 1000000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"

kwargs = {
    "en_nlp": en_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

train_data = [utils.tokenize_example(word, kwargs['en_nlp'], max_length, kwargs['sos_token'], kwargs['eos_token']) for word in train_set]
valid_data = [utils.tokenize_example(word, kwargs['en_nlp'], max_length, kwargs['sos_token'], kwargs['eos_token']) for word in valid_set]
test_data = [utils.tokenize_example(word, kwargs['en_nlp'], max_length, kwargs['sos_token'], kwargs['eos_token']) for word in test_set]

''' building our vocab'''
''' min_freq: argument to not create an index for tokens which appear less than min_freq'''
''' unk_token: all unknown tokens are replaced by <unk>'''
''' pad_token: add pad to every sentence have the same length'''
min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

special_tokens = [
    unk_token,
    pad_token,
    sos_token,
    eos_token,
]

en_data = [d['en_tokens'] for d in train_data]
machine_data = [d['machine_tokens'] for d in train_data]

en_vocab = torchtext.vocab.build_vocab_from_iterator(
    en_data,
    min_freq=min_freq,
    specials=special_tokens,
)

machine_vocab = torchtext.vocab.build_vocab_from_iterator(
    machine_data,
    min_freq=min_freq,
    specials=special_tokens,

)


'''here we'll programmatically get it and also check that both our vocabularies 
have the same index for the unknown and padding tokens.'''
assert en_vocab[unk_token] == machine_vocab[unk_token]
assert en_vocab[pad_token] == machine_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

'''we can set what value is returned when we try and
get the index of a token outside of our vocabulary. 
 In this case, the index of the unknown token'''
en_vocab.set_default_index(unk_index)
machine_vocab.set_default_index(unk_index)


'''We apply the numericalize_example function, 
passing our vocabularies in the fn_kwargs dictionary to the fn_kwargs argument.'''
kwargs = {"en_vocab": en_vocab, "machine_vocab": machine_vocab}

train_data = [utils.numericalize_example(word, kwargs['en_vocab'], kwargs['machine_vocab'] ) for word in train_data]
valid_data = [utils.numericalize_example(word, kwargs['en_vocab'], kwargs['machine_vocab']) for word in valid_data]
test_data = [utils.numericalize_example(word, kwargs['en_vocab'], kwargs['machine_vocab']) for word in test_data]

'''creating our dataloaders'''
batch_size = 32
train_data_loader = utils.get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = utils.get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = utils.get_data_loader(test_data, batch_size, pad_index)

'''model initialization'''
model_info = {
    'input_dim': len(en_vocab),
    'output_dim': len(machine_vocab),
    'encoder_embedding_dim': 300,
    'decoder_embedding_dim': 300,
    'hidden_dim' : 1024,
    'n_layers':1,
    'encoder_dropout':0.5,
    'decoder_dropout':0.5,
    'device': "cuda"
}

encoder = Encoder(
    model_info['input_dim'],
    model_info['encoder_embedding_dim'],
    model_info['hidden_dim'],
    model_info['n_layers'],
    model_info['encoder_dropout'],
)

decoder = Decoder(
    model_info['output_dim'],
    model_info['decoder_embedding_dim'],
    model_info['hidden_dim'],
    model_info['n_layers'],
    model_info['decoder_dropout'],
)

model = Seq2Seq(encoder, decoder, model_info['device']).to(model_info['device'])
model.apply(model_utils.init_weights)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
"we ignore the loss whenever the target token is a padding token."
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

'''training! '''
n_epochs = 100
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = model_utils.train_function(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        model_info['device'],
    )
    valid_loss = model_utils.evaluate_function(
        model,
        valid_data_loader,
        criterion,
        model_info['device'],
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

#model.load_state_dict(torch.load("tut1-model.pt"))

# ''' testing translating!'''
# sentence = "Go to Living_Room next find cigarette take it."

# translation = model_utils.translate_sentence(
#     sentence,
#     model,
#     en_nlp,
#     en_vocab,
#     machine_vocab,
#     lower,
#     sos_token,
#     eos_token,
#     model_info['device'],
# )

#print(translation)
