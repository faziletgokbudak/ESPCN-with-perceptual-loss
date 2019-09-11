import copy
import torch
import argparse
import torch.backends.cudnn as cudnn

from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from model import ESPCN
from datasets import Train, Validation
from utils import Feature_Extractor, AverageMeter, psnr_calculate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_set', type=str)
    parser.add_argument('--val_set', type=str)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=56)
    parser.add_argument('--loss_coeff', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ESPCN(num_channel=1, scale=args.scale).to(device)

    k = args.loss_coeff
    if k != 0:
        feat_ext = Feature_Extractor().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    train_set = Train(args.training_set, scale=args.scale, patch_size=args.patch_size)
    trainloader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_set = Validation(args.val_set)
    valloader = DataLoader(val_set, batch_size=1,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True)
    best_epoch = 0
    best_PSNR = 0.0
    loss_plot = []
    psnr_plot = []
    best_weights = copy.deepcopy(net.state_dict())


    for epoch in range(args.epoch):
        net.train()
        epoch_loss = AverageMeter()

        for data in trainloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = net(inputs)

            preds_3d = preds.repeat(1, 3, 1, 1)
            labels_3d = labels.repeat(1, 3, 1, 1)

            if k != 0:
                feature_preds = feat_ext(preds_3d)
                feature_labels = feat_ext(labels_3d)

                pixelwise_loss = criterion(preds, labels)
                feature_loss = criterion(feature_preds, feature_labels)
                loss = pixelwise_loss + k*feature_loss
            else:
                loss = criterion(preds, labels)

            epoch_loss.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        epoch_PSNR = AverageMeter()

        for data in valloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = net(inputs)

            mse = criterion(preds, labels)
            epoch_PSNR.update(psnr_calculate(mse), len(inputs))

        print('epoch:', epoch, 'loss: {:.6f}'.format(epoch_loss.avg), 'PSNR: {:.2f}'.format(epoch_PSNR.avg))

        if epoch_PSNR.avg > best_PSNR:
            best_epoch = epoch
            best_PSNR = epoch_PSNR.avg
            best_weights = copy.deepcopy(net.state_dict())

        loss_plot.append(epoch_loss.avg)
        psnr_plot.append(epoch_PSNR.avg)

    # torch.save(best_weights, 'best_path.pth')
    print('best PSNR: {:.2f}'.format(best_PSNR), 'best epoch: {}'.format(best_epoch))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(len(loss_plot)), loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(len(psnr_plot)), psnr_plot)
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (in dB)')
    plt.show()
