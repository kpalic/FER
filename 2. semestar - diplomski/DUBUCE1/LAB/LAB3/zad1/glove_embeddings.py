import numpy as np
import torch

def load_glove_embeddings(glove_path, vocab):
    embedding_dim = 300
    embeddings = np.random.randn(len(vocab.itos), embedding_dim).astype(np.float32)
    embeddings[0] = np.zeros(embedding_dim)

    with open(glove_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            token = parts[0]
            if token in vocab.stoi:
                embeddings[vocab.stoi[token]] = np.array(parts[1:], dtype=np.float32)

    return torch.nn.Embedding.from_pretrained(torch.tensor(embeddings), padding_idx=0)
