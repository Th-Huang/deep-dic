import numpy as np
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from PIL import Image
from model import FeatureResNet, SegResNet
from scipy.io import savemat

root_result = '../output/'
model_result = root_result + 'model/'
log_result = root_result + 'log/'
first_image = '../DIC-dataset/test/cropped/t1-00000000_0.tif'
next_img = '../DIC-dataset/test/cropped/t1-00000001_0.tif'

def default_loader(path):
    return Image.open(path)

fnet = FeatureResNet()
fcn = SegResNet(2,fnet)
fcn = fcn.cuda()
fcn.load_state_dict(torch.load(model_result + 'param_all_2_99_1333'))

x1, x2, y1, y2 = 1380, 2556, 968, 1976  # ylo,yhi,xlo,xhi
x100, x200, y100, y200 = 1380, 2556, 968, 1976  # ylo,yhi,xlo,xhi
dx1, dx2, dy1, dy2 = 0, 0, 0, 0

h0 = x2 - x1
w0 = y2 - y1

disp_1_x = np.zeros((128, 128))
disp_1_y = np.zeros((128, 128))

path_img = ''
results_path = ''

img_num = 2  # Number of images to process

for i in range(1, img_num):
    h0 = x2 - x1
    w0 = y2 - y1
    hnew = int((h0 // 32 + 1) * 32)
    wnew = int((w0 // 32 + 1) * 32)
    newsize = (wnew, hnew)

    img1 = default_loader(first_image)
    img1_c = img1.crop((y1, x1, y2, x2))
    img1_r = img1_c.resize(newsize)
    img2 = default_loader(next_img)
    img2_c = img2.crop((y1, x1, y2, x2))
    img2_r = img2_c.resize(newsize)

    img_1 = ToTensor()(img1_r)
    img_2 = ToTensor()(img2_r)
    imgs = torch.cat((img_1 / np.max(img_1.numpy()), img_2 / np.max(img_2.numpy())), 0)
    imgs = imgs.unsqueeze(0)
    imgs = imgs.type(torch.cuda.FloatTensor)
    imgs = Variable(imgs).cuda()

    predict = fcn(imgs,mode='displacement')
    predict_np = predict.detach().cpu().numpy().squeeze(0)

    dy1 = dy1 + np.mean(predict_np[0, :, :]) * (w0) / (wnew) / 2.0
    dy2 = dy2 + np.mean(predict_np[0, :, :]) * (w0) / (wnew) / 2.0
    dx1 = dx1 + np.mean(predict_np[1, :, :]) * (h0) / (hnew) / 2.0
    dx2 = dx2 + np.mean(predict_np[1, :, :]) * (h0) / (hnew) / 2.0

    x10, x20, y10, y20 = x1, x2, y1, y2
    h0 = x20 - x10
    w0 = y20 - y10

    x1 = int(x100 - dx1)
    y1 = int(y100 - dy1)
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