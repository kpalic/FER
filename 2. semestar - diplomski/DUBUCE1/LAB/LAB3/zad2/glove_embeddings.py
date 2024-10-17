# glove_embeddings.py

import torch
import numpy as np

def load_glove_embeddings(glove_file, vocab):
    embedding_dim = 300  # GloVe embedding dimenzija
    embeddings = np.random.normal(size=(len(vocab.itos), embedding_dim))  # Inicijaliziraj nasumično

    embeddings[vocab.stoi['<PAD>']] = np.zeros(embedding_dim)  # Inicijaliziraj <PAD> na vektor nula

    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            if word in vocab.stoi:
                embeddings[vocab.stoi[word]] = vector

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    embedding_layer = torch.nn.Embedding.from_pretrained(embeddings, padding_idx=vocab.stoi['<PAD>'], freeze=True)
    
    return embedding_layer
