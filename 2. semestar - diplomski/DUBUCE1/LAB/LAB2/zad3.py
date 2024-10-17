import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # "conv1"
            nn.ReLU(),
            nn.MaxPool2d(2),  # "pool1"
            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # "conv2"
            nn.ReLU(),
            nn.MaxPool2d(2),  # "pool2"
            nn.Flatten(),  # "flatten3"
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),  # "fc3"
            nn.ReLU(),
            nn.Linear(512, 10),  # "logits"
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Konfiguracija
config = {
    'max_epochs': 8,
    'batch_size': 50,
    'lr': 1e-1,
    'weight_decay': 1e-3,  # Lambda za L2 regularizaciju
}

# DataLoader i transformacije
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# Instanciranje modela, gubitka i optimizatora
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

# Treniranje
for epoch in range(config['max_epochs']):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
