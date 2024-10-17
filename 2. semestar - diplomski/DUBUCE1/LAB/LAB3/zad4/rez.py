import pandas as pd

# Load the CSV file
df = pd.read_csv('hyperparameter_tuning_results.csv')

# Sort by Test Accuracy in descending order
df_sorted = df.sort_values(by='Test Accuracy', ascending=False)

# Save the sorted dataframe to a new CSV file
sorted_csv_path = 'sorted_hyperparameter_tuning_results.csv'
df_sorted.to_csv(sorted_csv_path, index=False)