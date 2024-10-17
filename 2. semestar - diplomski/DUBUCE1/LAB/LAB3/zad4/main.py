import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import sys
import os

# Add the parent directory to the system path to import zad1 modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zad1.collate_fn import pad_collate_fn
from zad1.nlp_dataset import NLPDataset
from zad1.vocab import Vocab
from zad1.instance import Instance
from zad2.glove_embeddings import load_glove_embeddings
from model import initialize_model
from zad3.train import train, evaluate
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
    text_vocab = Vocab(dict(sorted_frequencies), min_freq=args.min_freq)
    label_vocab = Vocab({'negative': 0, 'positive': 1}, max_size=-1, min_freq=0, specials=False)

    # Load GloVe embeddings
    glove_path = '../sst_glove_6b_300d.txt'
    embeddings = load_glove_embeddings(glove_path, text_vocab).to(args.device)

    # Load datasets
    train_dataset, valid_dataset, test_dataset = load_dataset(args.train_file, args.valid_file, args.test_file, text_vocab, label_vocab)

    # Create DataLoader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size_train, shuffle=True, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi['<PAD>']))
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size_test, shuffle=False, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi['<PAD>']))
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_test, shuffle=False, collate_fn=lambda x: pad_collate_fn(x, pad_index=text_vocab.stoi['<PAD>']))

    # Initialize the model
    model = initialize_model(embeddings, args).to(args.device)

    # Define loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    # Lists to store the results for averaging
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    valid_f1s = []

    # Training and evaluation
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, args.device, args.clip)
        valid_loss, valid_accuracy, valid_f1, valid_cm = evaluate(model, valid_dataloader, criterion, args.device)
        
        # Store the results
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        valid_f1s.append(valid_f1)

    # Calculate averages
    avg_train_loss = np.mean(train_losses)
    avg_valid_loss = np.mean(valid_losses)
    avg_valid_accuracy = np.mean(valid_accuracies)
    avg_valid_f1 = np.mean(valid_f1s)

    # Test evaluation
    test_loss, test_accuracy, test_f1, test_cm = evaluate(model, test_dataloader, criterion, args.device)

   # Print selected parameters and average results
    print("Experiment Parameters and Results:")
    selected_params = [
        'rnn_type', 'dropout', 'hidden_size', 'num_layers', 'bidirectional',
    ]
    for param in selected_params:
        print(f"{param}: {getattr(args, param)}")
    print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}, Valid F1: {valid_f1:.2f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}, Test F1: {test_f1:.2f}")

    return avg_train_loss, avg_valid_loss, avg_valid_accuracy, avg_valid_f1, test_loss, test_accuracy, test_f1

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
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='device to use for training and evaluation')
    parser.add_argument('--rnn_type', type=str, choices=['RNN', 'GRU', 'LSTM'], default='GRU')
    parser.add_argument('--hidden_size', type=int, default=150, help='hidden size of RNN layers')
    parser.add_argument('--num_layers', type=int, default=2, help='number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--min_freq', type=int, default=1, help='minimum frequency for vocabulary')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping value')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD', 'RMSprop'], default='Adam', help='optimizer to use')
    parser.add_argument('--nonlinearity', type=str, choices=['ReLU', 'Tanh'], default='ReLU', help='nonlinearity function')
    parser.add_argument('--pooling', type=str, choices=['mean', 'max'], default='mean', help='pooling method for baseline')
    args = parser.parse_args()

    results = main(args)
    print(results)
