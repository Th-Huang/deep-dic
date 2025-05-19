import torch
import torch.utils.data as data_utils
from torchvision.transforms import ToTensor
from PIL import Image
import scipy.io as sio
import numpy as np
def default_loader(path):
    return Image.open(path)

class MyDataset(data_utils.Dataset):
    def __init__(self, dataset, transform=None, target_transform=None, loader=default_loader):

        self.imgs = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        label_x, label_y, label_z = self.imgs[index]
        img1 = self.loader(label_x)
        img_1 = ToTensor()(img1.resize((128, 128)))
        img2 = self.loader(label_y)
        img_2 = ToTensor()(img2.resize((128, 128)))
        imgs = torch.cat((img_1, img_2), 0)
        try:
            gt = sio.loadmat(label_z)['Disp_field_1'].astype(float)
        except KeyError:
            gt = sio.loadmat(label_z)['Disp_field_2'].astype(float)
        gt = gt[::2, ::2, :]
        gt = np.moveaxis(gt, -1, 0)
        return imgs, gt

    def __len__(self):
        return len(self.imgs)