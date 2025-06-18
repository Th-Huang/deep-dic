import torch
from model import FeatureResNet, SegResNet
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils
import os
from MyDataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def executeEpoch(EPOCH, loss_func, fcn, optimizer, train_loader, test_loader, writer, mode='train'):

    dataString = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')
    root_result = '../output/'
    if not os.path.exists(root_result):
        os.mkdir(root_result)
    model_result = root_result + 'model/'
    log_result = root_result + 'log/'
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

   # fcn.load_state_dict(torch.load(model_result + 'PATH_TO_PRETRAINED')) #comment this line if you start a new training
    for epoch in range(EPOCH):
        fcn.train()
        lE = 0.0
        for step, (img, dis, gt) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader

            img = Variable(img).cuda()
            gt = gt.float()
            gt = Variable(gt).cuda()
            output = fcn(img,mode='strain')  # cnn output
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

        lE = lE / step
        writer.add_scalar('Loss/train', lE, epoch)

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

def train():
    fnet = FeatureResNet()
    fcn = SegResNet(4, fnet)
    fcn = fcn.cuda()

    dataset_path = '../DIC-dataset/'

    test_set = []
    for i in range(4000):
        test_set.append((dataset_path + 'imgs3/train_image_' + str(i + 1) + '_1.png',
                         dataset_path + 'imgs3/train_image_' + str(i + 1) + '_2.png',
                         dataset_path + 'gt3/train_image_' + str(i + 1) + '.mat'))

    train_set = []
    for z in range(16000):
        train_set.append((dataset_path + 'imgs3/train_image_' + str(z + 1) + '_1.png',
                          dataset_path + 'imgs3/train_image_' + str(z + 1) + '_2.png',
                          dataset_path + 'gt3/train_image_' + str(z + 1) + '.mat'))

    EPOCH = 100  # train the training data n times, to save time, we just train 100 epoch
    BATCH_SIZE = 12
    print('BATCH_SIZE = ', BATCH_SIZE)
    LR = 0.001  # learning rate
    # root = './gdrive_northwestern/My Drive/dl_encoder/data/orig/orig'
    NUM_WORKERS = 0

    optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)  # optimize all cnn parameters
    # optimizer = torch.optim.SGD(cnn.parameters(), lr=LR, momentum=0.9)   # optimize all cnn parameters
    loss_func = nn.MSELoss()

    train_data = MyDataset(dataset=train_set)
    train_loader = data_utils.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,
                                         num_workers=NUM_WORKERS)

    test_data = MyDataset(dataset=test_set)
    test_loader = data_utils.DataLoader(dataset=test_data, batch_size=1)
    expPath = '../output/runs/Strain_train/'

    writer = SummaryWriter(expPath)

    executeEpoch(EPOCH, loss_func, fcn, optimizer, train_loader, test_loader, writer, mode='train')

if __name__ == '__main__':
    train()