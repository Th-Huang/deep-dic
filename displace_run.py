import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torch.nn import init

from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.transforms import ToTensor
import io
from torchvision import models, transforms
import torch.utils.data as data_utils
from PIL import Image
import os

import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
def default_loader(path):
    return Image.open(path)


from torchvision.models.resnet import BasicBlock, ResNet
from torch.nn import init


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1,
                                   output_padding=1,
                                   dilation=dilation, bias=bias)
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=bias)
    if bias:
        init.constant(layer.bias, 0)
    return layer


# Returns 2D batch normalisation layer
def bn(planes):
    layer = nn.BatchNorm2d(planes)
    # Use mean 0, standard deviation 1 init
    init.constant(layer.weight, 1)
    init.constant(layer.bias, 0)
    return layer


class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 14, 16, 3], 1000)
        self.conv_f = conv(2, 64, kernel_size=3, stride=1)
        self.ReLu_1 = nn.ReLU(inplace=True)
        self.conv_pre = conv(512, 1024, stride=2, transposed=False)
        self.bn_pre = bn(1024)

    def forward(self, x):
        x1 = self.conv_f(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x2 = self.maxpool(x)
        x = self.layer1(x2)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.ReLu_1(self.bn_pre(self.conv_pre(x5)))
        return x1, x2, x3, x4, x5, x6


class SegResNet(nn.Module):
    def __init__(self, num_classes, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        # self.conv3 = conv(1024,1024, stride=1, transposed=False)
        # self.bn3 = bn(1024)
        self.conv3_2 = conv(1024, 512, stride=1, transposed=False)
        self.bn3_2 = bn(512)
        self.conv4 = conv(512, 512, stride=2, transposed=True)
        self.bn4 = bn(512)
        self.conv5 = conv(512, 256, stride=2, transposed=True)
        self.bn5 = bn(256)
        self.conv6 = conv(256, 128, stride=2, transposed=True)
        self.bn6 = bn(128)
        self.conv7 = conv(128, 64, stride=2, transposed=True)
        self.bn7 = bn(64)
        self.conv8 = conv(64, 64, stride=2, transposed=True)
        self.bn8 = bn(64)
        self.conv9 = conv(64, 32, stride=2, transposed=True)
        self.bn9 = bn(32)
        self.convadd = conv(32, 16, stride=1, transposed=False)
        self.bnadd = bn(16)
        self.conv10 = conv(16, num_classes, stride=2, kernel_size=5)
        init.constant(self.conv10.weight, 0)  # Zero init

    def forward(self, x):
        x1, x2, x3, x4, x5, x6 = self.pretrained_net(x)
        x = self.relu(self.bn3_2(self.conv3_2(x6)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x + x4)))
        x = self.relu(self.bn7(self.conv7(x + x3)))
        x = self.relu(self.bn8(self.conv8(x + x2)))
        x = self.relu(self.bn9(self.conv9(x + x1)))
        x = self.relu(self.bnadd(self.convadd(x)))
        x = self.conv10(x)
        return x


fnet = FeatureResNet()
fcn = SegResNet(2,fnet)
fcn = fcn.cuda()
fcn.load_state_dict(torch.load(model_result + 'pretrained_displacementnet'))

from scipy.io import savemat

x1, x2, y1, y2 = 440, 806, 25, 215  # ylo,yhi,xlo,xhi
x100, x200, y100, y200 = 440, 806, 25, 215  # ylo,yhi,xlo,xhi
dx1, dx2, dy1, dy2 = 0, 0, 0, 0

h0 = x2 - x1
w0 = y2 - y1

disp_1_x = np.zeros((128, 128))
disp_1_y = np.zeros((128, 128))

path_img = ''
results_path = ''

for i in range(1, img_num):
    h0 = x2 - x1
    w0 = y2 - y1
    hnew = int((h0 // 32 + 1) * 32)
    wnew = int((w0 // 32 + 1) * 32)
    newsize = (wnew, hnew)

    img1 = default_loader('first_image')
    img1_c = img1.crop((y1, x1, y2, x2))
    img1_r = img1_c.resize(newsize)
    img2 = default_loader('next_img')
    img2_c = img2.crop((y1, x1, y2, x2))
    img2_r = img2_c.resize(newsize)

    img_1 = ToTensor()(img1_r)
    img_2 = ToTensor()(img2_r)
    imgs = torch.cat((img_1 / np.max(img_1.numpy()), img_2 / np.max(img_2.numpy())), 0)
    imgs = imgs.unsqueeze(0)
    imgs = imgs.type(torch.cuda.FloatTensor)
    imgs = Variable(imgs).cuda()

    predict = fcn(imgs)
    predict_np = predict.detach().cpu().numpy().squeeze(0)

    dy1 = dy1 + np.mean(predict_np[0, :, :]) * (w0) / (wnew) / 2.0
    dy2 = dy2 + np.mean(predict_np[0, :, :]) * (w0) / (wnew) / 2.0
    dx1 = dx1 + np.mean(predict_np[1, :, :]) * (h0) / (hnew) / 2.0
    dx2 = dx2 + np.mean(predict_np[1, :, :]) * (h0) / (hnew) / 2.0

    x10, x20, y10, y20 = x1, x2, y1, y2
    h0 = x20 - x10
    w0 = y20 - y10

    x1 = np.int(x100 - dx1)
    y1 = np.int(y100 - dy1)
    x2 = round(x200 - dx2)
    y2 = round(y200 - dy2)  ###new roi updated

    disp_1_x = predict_np / 2

    position = [x1, x2, y1, y2, h0, w0]

    matname = results_path + '/result_' + str(i) + '_position1.mat'
    mdic = {"position": position, "label": "position"}
    savemat(matname, mdic)

    matname = results_path + '/result_' + str(i) + '_disp_0_x.mat'
    mdic = {"disp_1_x": disp_1_x, "label": "disp_1_x"}
    savemat(matname, mdic)

