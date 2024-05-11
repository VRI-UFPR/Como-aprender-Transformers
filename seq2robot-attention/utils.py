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
from torch.utils import data

# =============================================================================
#  Functions
# =============================================================================

'''apply the tokenizer to all of the examples in data split 
and then appends the start of sequence and end of sequence tokens 
to the beginning and end of the list of tokens'''
def tokenize_example(example, en_nlp, max_length, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example[0])][:max_length]
    machine_words = str(example[1])
    machine_tokens = [token.text for token in en_nlp.tokenizer(machine_words)][:max_length]
    en_tokens = [token.lower() for token in en_tokens]
    machine_tokens = [token.lower() for token in machine_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    machine_tokens = [sos_token] + machine_tokens + [eos_token]

    return {"en_tokens": en_tokens, "machine_tokens": machine_tokens}

'''convert tokens to indices and returns tensors, so as an example:
"i", "love", "watching", "crime", "shows" -> [956, 2169, 173, 0, 821]'''
def numericalize_example(example, en_vocab, machine_vocab):
    en_ids = torch.tensor(en_vocab.lookup_indices(example['en_tokens']))
    machine_ids = torch.tensor(machine_vocab.lookup_indices(example['machine_tokens']))
    return {"en_ids": en_ids, "machine_ids":machine_ids}

def get_collate_fn(pad_index):
    def collate(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_machine_ids = [example["machine_ids"] for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_machine_ids = nn.utils.rnn.pad_sequence(batch_machine_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "machine_ids": batch_machine_ids,
        }
        return batch

    return collate

'''get_data_loader is created using a Dataset, the batch size, 
the padding token index (which is used for creating the batches 
in the collate_fn, and a boolean deciding if the examples should be 
shuffled at the time the data loader is iterated over.'''
def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


