import torch.nn as nn

class BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(BNReluConv, self).__init__()
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, padding=k // 2, bias=bias))