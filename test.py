import argparse
import os
import re
import glob
import numpy as np
import scipy.io as sio
# from vis_tools import Visualizer

import torch
import torch.nn as nn
import torch.optim as optim
import modelPAILaddproj

from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from skimage import metrics
import time
import sys
import scipy.io as scio

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")#2
parser.add_argument("--lr", type=float, default=2.5e-4, help="adam: learning rate")
parser.add_argument("--n_block", type=int, default=10)#40
parser.add_argument("--n_cpu", type=int, default=5)
parser.add_argument("--model_save_path", type=str, default="saved_PAILaddprojmodels/14th")
parser.add_argument('--checkpoint_interval', type=int, default=5)
opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False
imagesize = 256

class testset_loader(Dataset):
    def __init__(self):
        self.files_A = sorted(glob.glob('./test_data/' + '*.mat'))

    def __getitem__(self, index):
        file_A = self.files_A[index]
        input_data = scio.loadmat(file_A)['imagesparse']
        label_data = scio.loadmat(file_A)['imagef']
        label_data2 = scio.loadmat(file_A)['imagef2']
        prj_data = scio.loadmat(file_A)['data']

        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        label_data2 = torch.FloatTensor(label_data2).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)

        return input_data, label_data, prj_data, label_data2

    def __len__(self):
        return len(self.files_A)


class net():
    def __init__(self):
        self.model = modelPAILaddproj.PAIL(opt.n_block, views=36, dets=880, width=imagesize, height=imagesize,
            dImg=0.009*2, Ang0=0, dDet=0.0011, dAng=2*3.14159/2200*40, s2r=5.3852, d2r=0, binshift=-0.0013,scale=1, scan_type=1)
        self.loss = nn.MSELoss()
        self.path = opt.model_save_path
        self.test_data = DataLoader(testset_loader(),
                                    batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
        self.start = 0
        self.epoch = opt.epochs
        self.check_saved_model()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.8)
        if cuda:
            self.model = self.model.cuda()

    def check_saved_model(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.initialize_weights()
        else:
            model_list = glob.glob(self.path + '/PAILmodel_epoch_*.pth')
            if len(model_list) == 0:
                self.initialize_weights()
            else:
                last_epoch = 0
                for model in model_list:
                    epoch_num = int(re.findall(r'PAILmodel_epoch_(-?[0-9]\d*).pth', model)[0])
                    if epoch_num > last_epoch:
                        last_epoch = epoch_num
                self.start = last_epoch
                self.model.load_state_dict(torch.load(
                    '%s/PAILmodel_epoch_%04d.pth' % (self.path, last_epoch)))
                print('Loaded model:',
                    '%s/PAILmodel_epoch_%04d.pth' % (self.path, last_epoch))

    def displaywin(self, img):
        img[img < 0] = 0
        high = img.max()
        low = 0
        img = (img - low) / (high - low) * 255
        return img

    def initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, modelPAILaddproj.prj_module):
                nn.init.normal_(module.weight, mean=0.05, std=0.01)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()

            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def test(self):
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        class Logger(object):
            def __init__(self, filename="Default.log"):
                self.terminal = sys.stdout
                self.log = open(filename, "a")

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)

            def flush(self):
                pass

        path = os.path.abspath(os.path.dirname(__file__))
        type = sys.getfilesystemencoding()
        sys.stdout = Logger('test_result.txt')
        print(path)
        print('------------------')
        psnr1 = 0
        for batch_index, data in enumerate(self.test_data):
            input_data, label_data, prj_data, label_data2 = data
            if cuda:
                input_data = input_data.cuda()
                label_data = label_data.cuda()
                label_data2 = label_data2.cuda()
                prj_data = prj_data.cuda()

            with torch.no_grad():
                output, output2 = self.model(input_data, prj_data)
            res = output.cpu().numpy()
            res2 = output2.cpu().numpy()
            reference = label_data.cpu().numpy()
            reference2 = label_data2.cpu().numpy()
            inputs = input_data.cpu().numpy()
            output = (self.displaywin(output) / 255).view(-1, 256, 256).cpu().numpy()
            label = (self.displaywin(label_data) / 255).view(-1, 256, 256).cpu().numpy()
            psnr = np.zeros(output.shape[0])
            for i in range(output.shape[0]):
                count = (batch_index) * output.shape[0] + i
                psnr[i] = metrics.peak_signal_noise_ratio(label[i], output[i])
                print("count:%f, psnr: %f" % (count, psnr[i]))
                psnr1 = psnr1 + psnr[i]
                sio.savemat('result.mat', {'data': res[i, 0], 'data2': res2[i, 0], 'psnr': psnr[i],
                                      'reference': reference[i, 0], 'reference2': reference2[i, 0],
                                      'inputs': inputs[i, 0]})


if __name__ == "__main__":
    network = net()
    network.test()
