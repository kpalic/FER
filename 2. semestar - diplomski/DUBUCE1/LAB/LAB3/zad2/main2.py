import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
import os

# Add the parent directory to the system path to import zad1 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zad1.collate_fn import pad_collate_fn
from zad1.nlp_dataset import NLPDataset
from zad1.vocab import Vocab
from zad1.instance import Instance
from glove_embeddings import load_glove_embeddings
from model import initialize_model
from train import train, evaluate
from typing import Dict

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

def load_dataset(train_file, valid_file, test_file, text_vocab, label_vocab):
    train_dataset = NLPDataset.from_file(train_file, text_vocab, label_vocab)
    valid_dataset = NLPDataset.from_file(valid_file, text_vocab, label_vocab)
    test_dataset = NLPDataset.from_file(test_file, text_vocab, label_vocab)
    return train_dataset, valid_dataset, test_dataset

def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Build frequencies and vocabulary
    frequencies = build_frequencies(args.train_file)
    sorted_frequencies = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))
    text_vocab = Vocab(dict(sorted_frequencies), max_size=-1, min_freq=1)
    label_vocab = Vocab({'negative': 0, 'positive': 1}, max_size=-1, min_freq=0, specials=False)

    # Load GloVe embeddings
    glove_path = '../sst_glove_6b_300d.txt'
    embeddings = load_glove_embeddings(glove_path, text_vocab).to(args.device)


    # # Prikazivanje nekoliko embedding vektora
    # print(f"Embedding vector for 'the': {embeddings(torch.tensor([text_vocab.stoi['the']]).to(args.device)).cpu().detach().numpy()}")
    # print(f"Embedding vector for 'a': {embeddings(torch.tensor([text_vocab.stoi['a']]).to(args.device)).cpu().detach().numpy()}")
    # print(f"Embedding vector for 'film': {embeddings(torch.tensor([text_vocab.stoi['film']]).to(args.device)).cpu().detach().numpy()}")


    # Load datasets
    train_dataset, valid_dataset, test_dataset = load_dataset(args.train_file, args.valid_file, args.test_file, text_vocab, label_vocab)

    # Create DataLoader
     # Create DataLoader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size_train, shuffle=True, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi['<PAD>']))
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size_test, shuffle=False, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi['<PAD>']))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_test, shuffle=False, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi['<PAD>']))

    # Initialize the model
    model = initialize_model(embeddings, args).to(args.device)

    # Define loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training and evaluation
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, args.device)
        valid_loss, valid_accuracy, valid_f1, valid_cm = evaluate(model, valid_dataloader, criterion, args.device)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}, Valid F1: {valid_f1:.2f}")
        print(f"Valid Confusion Matrix:\n {valid_cm}")

    # Test evaluation
    test_loss, test_accuracy, test_f1, test_cm = evaluate(model, test_dataloader, criterion, args.device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}, Test F1: {test_f1:.2f}")
    print(f"Test Confusion Matrix:\n {test_cm}")

    return train_loss, valid_loss, valid_accuracy, valid_f1, test_loss, test_accuracy, test_f1

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Baseline Model for NLP Task")
    parser.add_argument('--seed', type=int, default=7052020, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size_train', type=int, default=32, help='batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=10, help='batch size for validation and testing')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--train_file', type=str, required=True, help='path to the training data file')
    parser.add_argument('--valid_file', type=str, required=True, help='path to the validation data file')
    parser.add_argument('--test_file', type=str, required=True, help='path to the test data file')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='device to use for training and evaluation')
    args = parser.parse_args()

    results = main(args)
    print(results)
