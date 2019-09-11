import torch
import numpy as np
from torch import nn
from torchvision import models


class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        pretrained_features = models.vgg16(pretrained=True).features
        self.layers = nn.Sequential()

        for i in range(9):
            self.layers.add_module(str(i), pretrained_features[i])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.layers(x)
        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def psnr_calculate(mse):
    return 10 * np.log10(1/mse.item())


def convert_rgb_to_y(img):
    return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.


def convert_rgb_to_ycbcr(img):
    y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
    cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img):
    r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
    g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
    b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def pre_process(img):
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    tensor_img = torch.from_numpy(img)
    return tensor_img.unsqueeze(0).unsqueeze(0)





