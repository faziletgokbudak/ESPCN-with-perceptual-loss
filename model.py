import torch.nn as nn
import torch.nn.functional as F

class ESPCN(nn.Module):
    def __init__(self, num_channel, scale):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 64, (5, 5), padding=5//2)
        self.conv2 = nn.Conv2d(64, 32, (3, 3), padding= 3//2)
        self.conv3 = nn.Conv2d(32, scale**2, (3, 3), padding=3//2)
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.weight_init()

    def weight_init(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv2.bias)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixelshuffle(x)
        return x
