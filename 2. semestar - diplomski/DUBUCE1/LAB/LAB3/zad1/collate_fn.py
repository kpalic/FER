# collate_fn.py

import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])

    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels_tensor = torch.tensor(labels)

    return texts_padded, labels_tensor, lengths
