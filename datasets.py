import glob
import numpy as np
import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset

class trainset_loader(Dataset):
    def __init__(self):
        # self.file_path = 'input_' + dose
        # self.files_A = sorted(glob.glob(os.path.join(root, 'train', self.file_path, 'data') + '*.mat'))
        self.files_A = sorted(glob.glob('/opt/data/private/pjy_TIP/训练+测试数据/AAPM/36sparseview/train/label/' + '*.mat'))
    def __getitem__(self, index):
        file_A = self.files_A[index]
        # file_B = file_A.replace(self.file_path,'label_single')
        # file_C = file_A.replace('input','proj24')
        file_B = file_A.replace('label','proj36')
        label_data = scio.loadmat(file_B)['imagef']
        label_data2 = scio.loadmat(file_B)['imagef2']
        input_data = scio.loadmat(file_B)['imagesparse']
        prj_data = scio.loadmat(file_B)['data']
        prj_label = scio.loadmat(file_B)['data1']
        
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        label_data2 = torch.FloatTensor(label_data2).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)
        prj_label = torch.FloatTensor(prj_label).unsqueeze_(0)
        
        return input_data, label_data, prj_data, label_data2, prj_label

    def __len__(self):
        return len(self.files_A)

class testset_loader(Dataset):
    def __init__(self):
        self.files_A = sorted(glob.glob('/opt/data/private/pjy_TIP/训练+测试数据/AAPM/36sparseview/test/label/' + '*.mat'))
        
    def __getitem__(self, index):
        file_A = self.files_A[index]
        # file_B = file_A.replace(self.file_path,'label_single')
        # file_C = file_A.replace('input','proj24')
        file_B = file_A.replace('label','proj36')
        res_name = './/result//' + file_A[-9:]
        input_data = scio.loadmat(file_B)['imagesparse']
        label_data = scio.loadmat(file_B)['imagef']
        label_data2 = scio.loadmat(file_B)['imagef2']
        prj_data = scio.loadmat(file_B)['data']
               
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        label_data2 = torch.FloatTensor(label_data2).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)
        
        return input_data, label_data, prj_data, res_name,label_data2

    def __len__(self):
        return len(self.files_A)
