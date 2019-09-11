import torch
import numpy as np
import argparse
from PIL import Image
import torch.backends.cudnn as cudnn

from torch import nn
from model import ESPCN
from matplotlib import pyplot as plt
from utils import pre_process, psnr_calculate, convert_ycbcr_to_rgb, convert_rgb_to_ycbcr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--test_img', type=str)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    criterion = nn.MSELoss()
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = ESPCN(num_channel=1, scale=args.scale)

    net.load_state_dict(torch.load(args.weights, map_location=device))

    with torch.no_grad():
        net.eval()

    img = Image.open(args.test_img, mode='r').convert('RGB')
    height, weight = (img.size[0] // args.scale) * args.scale, (img.size[1] // args.scale) * args.scale

    lr = img.resize((height // args.scale, weight // args.scale), Image.BICUBIC)
    bicubic = lr.resize((height, weight), Image.BICUBIC)
    lr = pre_process(lr.convert('L')).to(device)

    tensor_sr = net(lr)
    img_y = np.array(img.convert('L')) / 255.0
    sr_y = tensor_sr.squeeze(0).squeeze(0).detach().numpy()

    ycbcr = convert_rgb_to_ycbcr(np.array(bicubic)) / 255.0
    sr_ycbcr = np.zeros((sr_y.shape[0], sr_y.shape[1], 3))
    sr_ycbcr[..., 0] = sr_y
    sr_ycbcr[..., 1:3] = ycbcr[..., 1:3]
    sr = convert_ycbcr_to_rgb(sr_ycbcr * 255.0) / 255.0

    mse = np.mean((img_y-sr_y)**2)
    PSNR = psnr_calculate(mse)

    print('PSNR: {:.2f}'.format(PSNR))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(bicubic)
    ax1.title.set_text('Bicubic')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(sr)
    ax2.title.set_text('Reconstructed')
    plt.show()

