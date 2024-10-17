import torch
import torch.nn as nn
from BNReluConv import BNReluConv
class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.conv1 = BNReluConv(input_channels, emb_size, k=3)
        self.conv2 = BNReluConv(emb_size, emb_size, k=3)
        self.conv3 = BNReluConv(emb_size, emb_size, k=3)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def get_features(self, img):
        # Vraća tenzor dimenzija BATCH_SIZE, EMB_SIZE
        x = self.conv1(img)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Osigurava da prva dimenzija bude veličina mini grupe
        return x

    def loss(self, anchor, positive, negative, margin=1.0, epsilon=1e-8):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # Implementacija trojnog gubitka
        pos_dist = torch.norm(a_x - p_x, p=2, dim=1) + epsilon
        neg_dist = torch.norm(a_x - n_x, p=2, dim=1) + epsilon
        loss = torch.mean(torch.max(pos_dist - neg_dist + margin, torch.zeros_like(pos_dist)))
        return loss


