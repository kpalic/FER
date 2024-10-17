import numpy as np
import torch
from torch.utils.data import DataLoader

dataset = [(torch.randn(4, 4), torch.randint(5, size=())) for _ in range(25)]
dataset = [(x.numpy(), y.numpy()) for x, y in dataset]
loader = DataLoader(dataset, batch_size=8, shuffle=False, 
                    num_workers=0, collate_fn=None, drop_last=False)
for x, y in loader:
  print(x.shape, y.shape)