from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
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
    'weight_decay': 1e-3,
}

# DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

# Inicijalizacija TensorBoarda
writer = SummaryWriter('runs/MNIST_experiment_1')

# Treniranje
for epoch in range(config['max_epochs']):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Zapisivanje gubitka
        if i % 100 == 0:  # svakih 100 koraka
            print(f'Epoch [{epoch+1}/{config["max_epochs"]}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
            writer.add_scalar('training_loss', loss.item(), epoch * len(train_loader) + i)

writer.close()
