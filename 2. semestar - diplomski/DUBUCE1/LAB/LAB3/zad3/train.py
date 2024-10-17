# train.py
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train(model, data, optimizer, criterion, device, clip_value):
    model.train()
    total_loss = 0
    for batch_num, (x, y, lengths) in enumerate(data):
        model.zero_grad()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y.float().unsqueeze(1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data)
    return avg_loss

def evaluate(model, data, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_num, (x, y, lengths) in enumerate(data):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y.float().unsqueeze(1))
            total_loss += loss.item()
            preds = torch.sigmoid(logits).round().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(data)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, accuracy, f1, cm
