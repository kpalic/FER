import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import compute_representations, evaluate

# Validation Accuracy: 80.71%

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # Spljošti sliku u vektor
        feats = img.view(img.size(0), -1)
        return feats

def main():
    # Definiramo transformacije koje će biti primijenjene na podatke
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Učitavanje trening skupa
    mnist_download_root = "./mnist/"
    trainset = MNIST(root=mnist_download_root, train=True, download=True, transform=transform)

    # Podjela trening skupa na trening i validacijski skup
    train_idx, val_idx = train_test_split(list(range(len(trainset))), test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(trainset, train_idx)
    val_subset = torch.utils.data.Subset(trainset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    # Učitavanje test skupa
    testset = MNIST(root=mnist_download_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    num_classes = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = IdentityModel().to(device)
    emb_size = 28 * 28  # Veličina spljoštene slike

    # Računanje reprezentacija za trening
    train_representations = compute_representations(model, train_loader, num_classes, emb_size, device)

    # Računanje reprezentacija za validacijski skup
    val_representations = compute_representations(model, val_loader, num_classes, emb_size, device)
    
    # Evaluacija na validacijskom skupu
    val_accuracy = evaluate(model, val_representations, val_loader, device)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
