# main.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from nlp_dataset import NLPDataset
from vocab import Vocab
from glove_embeddings import load_glove_embeddings
from collate_fn import pad_collate_fn

# Provjera dostupnosti GPU-a
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Kreiranje frekvencijskog rječnika
def build_frequencies(dataset):
    frequencies = {}
    for instance in dataset:
        for token in instance[0]:
            if token not in frequencies:
                frequencies[token] = 0
            frequencies[token] += 1
    return frequencies

# Učitavanje skupa podataka bez numerizacije
train_dataset_raw = NLPDataset.from_file('sst_train_raw.csv', None)
valid_dataset_raw = NLPDataset.from_file('sst_valid_raw.csv', None)
test_dataset_raw = NLPDataset.from_file('sst_test_raw.csv', None)

# Izgradnja frekvencija
frequencies = build_frequencies(train_dataset_raw)

# Izgradnja vokabulara
text_vocab = Vocab(frequencies, max_size=-1, min_freq=0)

# Učitavanje skupa podataka s numerizacijom
train_dataset = NLPDataset.from_file('sst_train_raw.csv', text_vocab)
valid_dataset = NLPDataset.from_file('sst_valid_raw.csv', text_vocab)
test_dataset = NLPDataset.from_file('sst_test_raw.csv', text_vocab)

# Učitavanje GloVe vektora
glove_path = 'sst_glove_6b_300d.txt'  # Promijenite putanju do GloVe datoteke ako je potrebno
embedding = load_glove_embeddings(glove_path, text_vocab).to(device)

# Inicijalizacija DataLoader-a
batch_size = 32
shuffle = True
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

# Definiranje jednostavnog modela
class SimpleTextClassifier(nn.Module):
    def __init__(self, embedding):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, 2)  # 2 za bidirectional
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        out = self.fc(h_n)
        return self.softmax(out)

# Instanciranje modela
model = SimpleTextClassifier(embedding).to(device)

# Definiranje kriterija gubitka i optimizatora
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Funkcija za trening modela
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for texts, labels, lengths in train_loader:
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
    accuracy = correct / len(train_loader.dataset)
    return total_loss / len(train_loader), accuracy

# Funkcija za evaluaciju modela
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for texts, labels, lengths in val_loader:
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(val_loader.dataset)
    return total_loss / len(val_loader), accuracy

# Trening i evaluacija modela
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_model(model, train_dataloader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate_model(model, valid_dataloader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Evaluacija modela na testnom skupu podataka
test_loss, test_accuracy = evaluate_model(model, test_dataloader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
