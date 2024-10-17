import argparse
import torch
import numpy as np
from main2 import main as run_model
import pandas as pd

def run_experiments(args, seeds):
    results = []

    for seed in seeds:
        args.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(seed)

        result = run_model(args)
        results.append(result)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=["Train Loss", "Valid Loss", "Valid Accuracy", "Valid F1", 
                                        "Test Loss", "Test Accuracy", "Test F1"])
    
    # Calculate mean of each metric
    mean_results = df.mean()
    
    # Save results to Excel
    df.to_csv("experiment_results.csv", index=False)
    
    return mean_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running multiple experiments with different seeds.")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_test', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()

    # List of different seeds
    seeds = [7052020, 1234567, 89101112, 13141516, 17181920]
    
    mean_results = run_experiments(args, seeds)
    
    print("Mean Results:")
    print(mean_results)
