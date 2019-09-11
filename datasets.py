import h5py
import random
import numpy as np
from utils import convert_rgb_to_y
from torch.utils.data import Dataset


class Train(Dataset):
    def __init__(self, training_set, scale, patch_size):
        super(Train, self).__init__()
        self.training_set = training_set
        self.patch_size = patch_size
        self.scale = scale

    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[1] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[0] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def __getitem__(self, idx):
        with h5py.File(self.training_set, 'r') as f:
            lr = f['lr'][str(idx)][::]
            hr = f['hr'][str(idx)][::]
            lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
            lr, hr = self.random_horizontal_flip(lr, hr)
            lr, hr = self.random_vertical_flip(lr, hr)
            lr, hr = self.random_rotate_90(lr, hr)
            lr = convert_rgb_to_y(lr)
            hr = convert_rgb_to_y(hr)
            lr = np.expand_dims(lr.astype(np.float32), 0) / 255.0
            hr = np.expand_dims(hr.astype(np.float32), 0) / 255.0
            return lr.astype(np.float32), hr.astype(np.float32)

    def __len__(self):
        with h5py.File(self.training_set, 'r') as f:
            return len(f['lr'])


class Validation(Dataset):
    def __init__(self, val_set):
        super(Validation, self).__init__()
        self.val_set = val_set

    def __getitem__(self, idx):
        with h5py.File(self.val_set, 'r') as f:
            lr = f['lr'][str(idx)][::]
            hr = f['hr'][str(idx)][::]
            lr = convert_rgb_to_y(lr)
            hr = convert_rgb_to_y(hr)
            lr = np.expand_dims(lr.astype(np.float32), 0) / 255.0
            hr = np.expand_dims(hr.astype(np.float32), 0) / 255.0
            return lr.astype(np.float32), hr.astype(np.float32)

    def __len__(self):
        with h5py.File(self.val_set, 'r') as f:
            return len(f['lr'])
