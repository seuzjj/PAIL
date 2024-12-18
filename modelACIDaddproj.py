import torch
import torch.nn as nn
import math
from torch.autograd import Function
import torchvision.models as models
import ctlib
from unet import UNet2D
# import graph_laplacian
ata=1.5624
#---------------------------------------------------------------------
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale=2, num_feat=64):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
#---------------------------------------------------------------------------------------


class prj_module(nn.Module):
    def __init__(self, options):
        super(prj_module, self).__init__()  
        self.weight = nn.Parameter(torch.Tensor(1))
        self.options = nn.Parameter(options, requires_grad=False)
        
    def forward(self, input_data, proj, lambda1, miu):        
        return prj_fun.apply(input_data,  proj, lambda1,  miu, self.options)

class prj_fun(Function):
    @staticmethod
    def forward(self, input_data, proj, lambda1, miu, options):
        
        temp = ( proj-ctlib.projection(input_data, options))/(ata+lambda1+miu)# ata=1.5624
        intervening_res = 2200/1440*ctlib.fbp(temp, options)
        self.save_for_backward(intervening_res, lambda1, miu,options)
        out =  intervening_res
        return out

    @staticmethod
    def backward(self, grad_output):
        intervening_res, lambda1,miu, options = self.saved_tensors
        temp = ctlib.projection(grad_output, options)
        temp = 2200/1440*ctlib.fbp(temp, options)
        grad_input =  -temp/(ata+lambda1+miu)
        temp = -intervening_res * grad_output/((ata+lambda1+miu)*(ata+lambda1+miu))
        grad_lambda1 =  temp.sum().view(-1)
        temp = -intervening_res * grad_output/((ata+lambda1+miu)*(ata+lambda1+miu))
        grad_miu =  temp.sum().view(-1)
        return grad_input, grad_lambda1, grad_miu, None, None

class prj_module0(nn.Module):
    def __init__(self, options):
        super(prj_module0, self).__init__()  
        self.weight = nn.Parameter(torch.Tensor(1))
        self.options = nn.Parameter(options, requires_grad=False)
        
    def forward(self,  proj):        
        return prj_fun0.apply(  proj, self.options)

class prj_fun0(Function):
    @staticmethod
    def forward(self, proj, options):
        intervening_res = ctlib.fbp(proj, options)
        self.save_for_backward(intervening_res, options)
        out =  intervening_res
        return out

    @staticmethod
    def backward(self, grad_output):
        intervening_res, options = self.saved_tensors
        temp = ctlib.projection(grad_output, options)
        
        grad_input =  temp
        return grad_input, None, None


class prj_module00(nn.Module):
    def __init__(self, options):
        super(prj_module00, self).__init__()  
        self.options = nn.Parameter(options, requires_grad=False)
        
    def forward(self, input_data):        
        return prj_fun00.apply(input_data, self.options)

class prj_fun00(Function):
    @staticmethod
    def forward(self, input_data, options):
        temp=ctlib.projection(input_data, options)
        intervening_res = temp
        self.save_for_backward(intervening_res, options)
        out =  intervening_res
        return out

    @staticmethod
    def backward(self, grad_output):
        intervening_res, options = self.saved_tensors
        temp = ctlib.fbp(grad_output, options)       
        grad_input =  temp


        return grad_input,None, None


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class IterBlocku(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(IterBlocku,self).__init__()
        numberchannel=8;
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=1,ch_out=numberchannel)
        self.Conv2 = conv_block(ch_in=numberchannel,ch_out=2*numberchannel)
        self.Conv3 = conv_block(ch_in=2*numberchannel,ch_out=4*numberchannel)
        self.Conv4 = conv_block(ch_in=4*numberchannel,ch_out=8*numberchannel)

     
        self.Up4 = up_conv(ch_in=8*numberchannel,ch_out=4*numberchannel)
        self.Up_conv4 = conv_block(ch_in=8*numberchannel, ch_out=4*numberchannel)
        
        self.Up3 = up_conv(ch_in=4*numberchannel,ch_out=2*numberchannel)
        self.Up_conv3 = conv_block(ch_in=4*numberchannel, ch_out=2*numberchannel)
        
        self.Up2 = up_conv(ch_in=2*numberchannel,ch_out=numberchannel)
        self.Up_conv2 = conv_block(ch_in=2*numberchannel, ch_out=numberchannel)

        self.Conv_1x1 = nn.Conv2d(numberchannel,1,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        output=d1

        return output
    
class IterBlockx(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(IterBlockx,self).__init__()
        numberchannel=8;
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=1,ch_out=numberchannel)
        self.Conv2 = conv_block(ch_in=numberchannel,ch_out=2*numberchannel)
        self.Conv3 = conv_block(ch_in=2*numberchannel,ch_out=4*numberchannel)
        self.Conv4 = conv_block(ch_in=4*numberchannel,ch_out=8*numberchannel)

     
        self.Up4 = up_conv(ch_in=8*numberchannel,ch_out=4*numberchannel)
        self.Up_conv4 = conv_block(ch_in=8*numberchannel, ch_out=4*numberchannel)
        
        self.Up3 = up_conv(ch_in=4*numberchannel,ch_out=2*numberchannel)
        self.Up_conv3 = conv_block(ch_in=4*numberchannel, ch_out=2*numberchannel)
        
        self.Up2 = up_conv(ch_in=2*numberchannel,ch_out=numberchannel)
        self.Up_conv2 = conv_block(ch_in=2*numberchannel, ch_out=numberchannel)

        self.Conv_1x1 = nn.Conv2d(numberchannel,1,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        output=d1+x

        return output
    
class IterBlock(nn.Module):
    def __init__(self, options):
        super(IterBlock, self).__init__()
        self.lambda1 = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True).cuda()
        self.miu = torch.nn.Parameter(torch.FloatTensor(1).cuda(), requires_grad=True).cuda()
        self.lambda1.data.fill_(3).cuda()
        self.miu.data.fill_(0).cuda()
        self.block1 = prj_module(options)
        self.block2 = IterBlocku()
        self.block3 = IterBlockx()
        self.block4 = UNet2D(in_channels=1, out_channels=1)
        

    def forward(self, input_data, proj):
        
        tmp1 = self.block1(input_data,proj,self.lambda1,self.miu)
        tmp2=self.block2(tmp1)
        temp3= input_data+((self.lambda1 + self.miu * 1) / ata) *tmp2
        
        output=self.block3(temp3)
        output1=self.block4(output)
        
          
        return output1

class ACID(nn.Module):
    def __init__(self, block_num, **kwargs):
        super(ACID, self).__init__()
        views = kwargs['views']
        dets = kwargs['dets']
        width = kwargs['width']
        height = kwargs['height']
        dImg = kwargs['dImg']
        dDet = kwargs['dDet']
        Ang0 = kwargs['Ang0']
        dAng = kwargs['dAng']
        s2r = kwargs['s2r']
        d2r = kwargs['d2r']
        binshift = kwargs['binshift']
        scan_type = kwargs['scan_type']
        options = torch.Tensor([views, dets, width, height, dImg, dDet, Ang0, dAng, s2r, d2r, binshift, scan_type])
        self.block = nn.ModuleList([IterBlock(options) for i in range(int(block_num))])
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(2, 16)
        self.conv_last = nn.Conv2d(16, 1, 3, 1, 1)    


    def forward(self, input_data, proj):
        x = input_data
        
        for index, module in enumerate(self.block):
            x = module(x, proj)
        output2 = self.conv_before_upsample(x)
        output2 = self.conv_last(self.upsample(output2))
        return x,output2
