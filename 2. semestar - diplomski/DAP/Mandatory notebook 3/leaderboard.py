import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold, ParameterGrid

# Učitavanje podataka
train_df = pd.read_csv('/kaggle/input/leaderboard-dataset/train_clean.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df = pd.read_csv('/kaggle/input/leaderboard-dataset/train_clean.csv')
test_df['Date'] = pd.to_datetime(test_df['Date'])

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.values

# Dataset i DataLoader klase
class StockDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test)

# Definicija različitih modela

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUPredictor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h_0)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

class HybridPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(HybridPredictor, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32 * input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), 1, -1)
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# Funkcija za treniranje modela
def train_model(train_loader, model, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{total_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    return running_loss / len(train_loader)

# Funkcija za evaluaciju modela
def evaluate_model(test_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(test_loader)

# Grid search
param_grid = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'epochs': [20, 50, 100],
    'model_type': ['LSTM', 'GRU', 'Hybrid']
}

best_params = None
best_loss = float('inf')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for params in ParameterGrid(param_grid):
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    epochs = params['epochs']
    model_type = params['model_type']
    
    fold_losses = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f'Starting fold {fold+1} for params {params}')
        train_subset = Subset(train_dataset, train_index)
        val_subset = Subset(train_dataset, val_index)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        if model_type == 'LSTM':
            model = LSTMPredictor(X_train.shape[1], hidden_size, num_layers).to(device)
        elif model_type == 'GRU':
            model = GRUPredictor(X_train.shape[1], hidden_size, num_layers).to(device)
        elif model_type == 'Hybrid':
            model = HybridPredictor(X_train.shape[1], hidden_size, num_layers).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            train_loss = train_model(train_loader, model, criterion, optimizer, device, epoch, epochs)
        
        val_loss = evaluate_model(val_loader, model, criterion, device)
        fold_losses.append(val_loss)
    
    avg_val_loss = np.mean(fold_losses)
    
    print(f'Params: {params}, Avg Validation Loss: {avg_val_loss}')
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_params = params

print('Best parameters found: ', best_params)
print('Best validation loss: ', best_loss)

# Treniranje najboljeg modela nad cijelim treniranjem skupom
best_hidden_size = best_params['hidden_size']
best_num_layers = best_params['num_layers']
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']
best_epochs = best_params['epochs']
best_model_type = best_params['model_type']

train_loader = DataLoader(train_dataset, best_batch_size, shuffle=True)

if best_model_type == 'LSTM':
    best_model = LSTMPredictor(X_train.shape[1], best_hidden_size, best_num_layers).to(device)
elif best_model_type == 'GRU':
    best_model = GRUPredictor(X_train.shape[1], best_hidden_size, best_num_layers).to(device)
elif best_model_type == 'Hybrid':
    best_model = HybridPredictor(X_train.shape[1], best_hidden_size, best_num_layers).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)

for epoch in range(best_epochs):
    train_loss = train_model(train_loader, best_model, criterion, optimizer, device, epoch, best_epochs)

# Predikcija nad testnim skupom
best_model.eval()
predictions = []
with torch.no_grad():
    for inputs in DataLoader(test_dataset, best_batch_size, shuffle=False):
        inputs = inputs.to(device)
        outputs = best_model(inputs)
        predictions.extend(outputs.cpu().numpy())

# Generiranje submission filea
submission_df = pd.DataFrame({'Id': np.arange(len(predictions)), 'Prediction': predictions})
