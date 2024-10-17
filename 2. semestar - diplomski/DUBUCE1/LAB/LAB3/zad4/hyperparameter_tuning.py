import argparse
import torch
import numpy as np
import pandas as pd
from main import main as run_model
from itertools import product

# Define the grid of hyperparameters
grid = {
    'rnn_type': ['RNN', 'GRU', 'LSTM'],
    # 'lr': [1e-3, 1e-4, 1e-5],
    'dropout': [0.0, 0.3, 0.5],
    'hidden_size': [100, 150, 200],
    'num_layers': [1, 2, 3],
    'bidirectional': [False, True],
    # 'min_freq': [1, 5, 10],
    # 'clip': [0.1, 0.25, 0.5, 0.75],
    # 'optimizer': ['Adam', 'SGD', 'RMSprop'],
    # 'nonlinearity': ['ReLU', 'Tanh'],
    # 'pooling': ['mean', 'max']
}

# Function to create all combinations of hyperparameters
def create_combinations(grid):
    keys = grid.keys()
    values = (grid[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    return combinations

def run_experiments(args, hyperparams, seeds):
    results = []

    for hp in hyperparams:
        for key, value in hp.items():
            setattr(args, key, value)
        
        for seed in seeds:
            args.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            if args.device == 'cuda':
                torch.cuda.manual_seed(seed)

            result = run_model(args)
            results.append({**hp, **dict(zip(["Train Loss", "Valid Loss", "Valid Accuracy", "Valid F1", 
                                              "Test Loss", "Test Accuracy", "Test F1"], result))})
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results to CSV
    df.to_csv("hyperparameter_tuning_results.csv", index=False)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running multiple experiments with different hyperparameters and seeds.")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size_train', type=int, default=32, help='batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=10, help='batch size for validation and testing')
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

    # List of different seeds
    seeds = [17181920]
    
    # Create all combinations of hyperparameters
    hyperparams = create_combinations(grid)
    
    # Run experiments
    df = run_experiments(args, hyperparams, seeds)
    
    print("Results saved to hyperparameter_tuning_results.csv")
