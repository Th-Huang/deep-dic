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
import scipy.io as sio

import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
def default_loader(path):
    return Image.open(path)

from torchvision.models.resnet import BasicBlock, ResNet
from torch.nn import init

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1,
                                   dilation=dilation, bias=bias)
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
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
        self.conv_f = conv(2,64, kernel_size=3,stride = 1)
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
        return x1, x2, x3, x4, x5,x6

class SegResNet(nn.Module):
    def __init__(self, num_classes, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        #self.conv3 = conv(1024,1024, stride=1, transposed=False)
        #self.bn3 = bn(1024)
        self.conv3_2 = conv(1024, 512, stride=1, transposed=False)
        self.bn3_2 = bn(512)
        self.conv4 = conv(512,512, stride=2, transposed=True)
        self.bn4 = bn(512)
        self.conv5 = conv(512, 256, stride=2, transposed=True)
        self.bn5 = bn(256)
        self.conv6 = conv(256, 128, stride=2, transposed=True)
        self.bn6 = bn(128)
        self.conv7 = conv(128, 64, stride=2, transposed=True)
        self.bn7 = bn(64)
        self.conv8 = conv(64, 64, stride=2, transposed=True)
        self.bn8 = bn(64)
        self.conv9 = conv(64, 32, stride=1, transposed=False)
        self.bn9 = bn(32)
        self.convadd = conv(32, 16, stride=1, transposed=False)
        self.bnadd = bn(16)
        self.conv10 = conv(16, num_classes,stride=2, kernel_size=3)
        init.constant(self.conv10.weight, 0)  # Zero init

    def forward(self, x):
        #b,c,w,h = x.size()
        #x = x.view(b,c,w,h)
        x1, x2, x3, x4, x5, x6 = self.pretrained_net(x)
        x = self.relu(self.bn3_2(self.conv3_2(x6)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x+x4 )))
        x = self.relu(self.bn7(self.conv7(x+x3 )))
        x = self.relu(self.bn8(self.conv8(x+x2 )))
        x = self.relu(self.bn9(self.conv9(x+x1 )))
        x = self.relu(self.bnadd(self.convadd(x)))
        x = self.conv10(x)
        return x
fnet = FeatureResNet()
fcn = SegResNet(4,fnet)
fcn = fcn.cuda()

dataset_path = '../DIC-dataset/'

test_set = []
for i in range(4000):
    test_set.append((dataset_path+'imgs3/train_image_'+str(i+1)+'_1.png',
                       dataset_path+'imgs3/train_image_'+str(i+1)+'_2.png',
                       dataset_path+'gt3/train_image_'+str(i+1)+'.mat'))

train_set = []
for z in range(16000):
    train_set.append((dataset_path+'imgs3/train_image_'+str(z+1)+'_1.png',
                       dataset_path+'imgs3/train_image_'+str(z+1)+'_2.png',
                       dataset_path+'gt3/train_image_'+str(z+1)+'.mat'))
import scipy.io as sio
from scipy import interpolate
x = np.arange(0,256,1)
y = np.arange(0,256,1)
xnew = np.arange(1.5,257.5,4)
ynew = np.arange(1.5,257.5,4)

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
EPOCH = 100              # train the training data n times, to save time, we just train 100 epoch
BATCH_SIZE = 12
print('BATCH_SIZE = ',BATCH_SIZE)
LR = 0.001              # learning rate
#root = './gdrive_northwestern/My Drive/dl_encoder/data/orig/orig'
NUM_WORKERS = 0

optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)   # optimize all cnn parameters
#optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9)   # optimize all cnn parameters
loss_func = nn.MSELoss()

train_data=MyDataset(dataset=train_set)
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

test_data=MyDataset(dataset=test_set)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=1)


from datetime import datetime
dataString = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')

root_result = '../output/'
if not os.path.exists(root_result):
    os.mkdir(root_result)
model_result = root_result+'model/'
log_result = root_result+'log/'
if not os.path.exists(model_result):
    os.mkdir(model_result)

if not os.path.exists(log_result):
    os.mkdir(log_result)

fileOut = open(log_result + 'log' + dataString, 'a')
fileOut.write(dataString + 'Epoch:   Step:    Loss:        Val_Accu :\n')
fileOut.close()
fileOut2 = open(log_result + 'validation' + dataString, 'a')
fileOut2.write('kernal_size of conv_f is 2')
fileOut2.write(dataString + 'Epoch:    loss:')

optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)  # optimize all cnn parameters

# fcn.load_state_dict(torch.load(model_result + 'PATH_TO_PRETRAINED')) #comment this line if you start a new training
for epoch in range(EPOCH):
    fcn.train()
    for step, (img, gt) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

        img = Variable(img).cuda()
        gt = gt.float()
        gt = Variable(gt).cuda()
        output = fcn(img)  # cnn output
        loss = loss_func(output, gt)  # loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        print(epoch, step, loss.data.item())
        fileOut = open(log_result + 'log' + dataString, 'a')
        fileOut.write(str(epoch) + '   ' + str(step) + '   ' + str(loss.data.item()) + '\n')
        fileOut.close()
    if epoch % 10 == 9:
        PATH = model_result + 'param_all_strain2_' + str(epoch) + '_' + str(step)
        torch.save(fcn.state_dict(), PATH)
        print('finished saving checkpoints')

    LOSS_VALIDATION = 0
    fcn.eval()
    with torch.no_grad():
        for step, (img, gt) in enumerate(test_loader):
            img = Variable(img).cuda()
            gt = gt.unsqueeze(1)  # batch x
            gt = Variable(gt).cuda()
            output = fcn(img)
            LOSS_VALIDATION += loss_func(output, gt)
        LOSS_VALIDATION = LOSS_VALIDATION / step
        fileOut2 = open(log_result + 'validation' + dataString, 'a')
        fileOut2.write(str(epoch) + '   ' + str(step) + '   ' + str(LOSS_VALIDATION.data.item()) + '\n')
        fileOut2.close()
        print('validation error epoch  ' + str(epoch) + ':    ' + str(LOSS_VALIDATION) + '\n' + str(step))