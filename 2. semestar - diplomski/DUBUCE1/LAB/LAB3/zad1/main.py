# main.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from nlp_dataset import NLPDataset
from vocab import Vocab
from glove_embeddings import load_glove_embeddings
from collate_fn import pad_collate_fn
from instance import Instance
from typing import List, Dict

# Provjera dostupnosti GPU-a
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Kreiranje frekvencijskog rječnika
def build_frequencies(filepath: str) -> Dict[str, int]:
    frequencies = {}
    with open(filepath, 'r') as file:
        for line in file:
            text, _ = line.strip().split(',')
            for token in text.split():
                if token not in frequencies:
                    frequencies[token] = 0
                frequencies[token] += 1
    return frequencies

# Izgradnja frekvencija
train_file_path = 'sst_train_raw.csv'
frequencies = build_frequencies(train_file_path)

# sortiranje frekvencija po vrijednosti, ako su iste vrijednosti, sortiraj po ključu (abecedno) silazno
# sorted_frequencies = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))

# most_common = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
# print("5 najčešćih riječi: ", most_common)

# Izgradnja vokabulara
text_vocab = Vocab(frequencies, max_size=-1, min_freq=0)
label_vocab = Vocab({'negative': 1, 'positive': 2}, max_size=-1, min_freq=0, specials=False)

# Ispisivanje vokabulara prvih 20 riječi
print("First 20 words in vocab: ")
print(text_vocab.itos[:20])

# Ispisivanje vokabulara prvih 20 labela
print("Label vocab: ")
print(label_vocab.itos)
print(label_vocab.stoi)

# Učitavanje skupa podataka s numerizacijom
train_dataset = NLPDataset.from_file('sst_train_raw.csv', text_vocab, label_vocab)
valid_dataset = NLPDataset.from_file('sst_valid_raw.csv', text_vocab, label_vocab)
test_dataset = NLPDataset.from_file('sst_test_raw.csv', text_vocab, label_vocab)

#print the first 5 instances from the training dataset
# print("First instance from the training dataset: ")
# print(train_dataset.instances[:1], "\n")


# Ispisivanje nekoliko primjera iz skupa podataka
print("Primjer iz trening skupa podataka: prvih 20")
# for i in range(20):
#     instance_text, instance_label = train_dataset.instances[i]
#     print(f"Text: {instance_text}")
#     print(f"Label: {instance_label}")
#     print(f"Numericalized text: {text_vocab.encode(instance_text)}")
#     print(f"Numericalized label: {label_vocab.encode(instance_label)}")

instance_text, instance_label = train_dataset.instances[3]
print(f"Text: {instance_text}")
print(f"Label: {instance_label}")
print(f"Numericalized text: {text_vocab.encode(instance_text)}")
print(f"Numericalized label: {label_vocab.encode(instance_label)}")

# Učitavanje GloVe vektora
glove_path = 'sst_glove_6b_300d.txt'  # Promijenite putanju do GloVe datoteke ako je potrebno
embedding = load_glove_embeddings(glove_path, text_vocab).to(device)

# Inicijalizacija DataLoader-a
batch_size = 2
shuffle = True
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

# Dohvaćanje batcha
texts, labels, lengths = next(iter(train_dataloader))
print("\nPrimjeri batcha iz trening skupa podataka:")
print(f"Texts: {texts}")
print(f"Labels: {labels}")
print(f"Lengths: {lengths}")

