import torch
import torch.utils.data as data_utils
from torchvision.transforms import ToTensor
import scipy.io as sio
import numpy as np
from scipy import interpolate
from PIL import Image

x = np.arange(0,256,1)
y = np.arange(0,256,1)
xnew = np.arange(1.5,257.5,4)
ynew = np.arange(1.5,257.5,4)


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
        img_1 = ToTensor()(img1.resize((128,128)))
        img2 = self.loader(label_y)
        img_2 = ToTensor()(img2.resize((128,128)))
        imgs = torch.cat((img_1, img_2), 0)
        try:
            gt = sio.loadmat(label_z)['Disp_field_1'].astype(float)

        except KeyError:
            gt = sio.loadmat(label_z)['Disp_field_2'].astype(float)

        gt = np.asarray(gt)
        gt = gt*100
        [dudx, dudy]= np.gradient(gt[:,:,0])
        [dvdx, dvdy]= np.gradient(gt[:,:,1])

        f = interpolate.interp2d(x, y, dudx, kind='cubic')
        dudx_ = f(xnew, ynew)
        f = interpolate.interp2d(x, y, dudy, kind='cubic')
        dudy_ = f(xnew, ynew)
        f = interpolate.interp2d(x, y, dvdx, kind='cubic')
        dvdx_ = f(xnew, ynew)
        f = interpolate.interp2d(x, y, dvdy, kind='cubic')
        dvdy_ = f(xnew, ynew)
        st = np.stack([dudx_, dudy_, dvdx_, dvdy_], axis=0)
                #st = np.stack([dudx, dudy, dvdx, dvdy], axis=0)

        return imgs,st

    def __len__(self):
        return len(self.imgs)