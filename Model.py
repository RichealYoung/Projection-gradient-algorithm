import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import psnr as cal_psnr
import cv2
from os.path import join as opj
import scipy.io
import scipy.stats as stats
import matplotlib.pyplot as plt
class conv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(conv, self).__init__()
        self.c = nn.Sequential(nn.Conv2d(inC, outC, kernel_size=kernel_size, padding=padding),
                               nn.BatchNorm2d(outC, momentum=momentum),
                               nn.ReLU())    
    def forward(self, x):
        x = self.c(x)
        return x

class convT(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, stride, momentum):
        super(convT, self).__init__()
        self.cT = nn.Sequential(nn.ConvTranspose2d(inC, outC, kernel_size=kernel_size, 
                                                   padding=padding,
                                                   stride=stride),
                                nn.BatchNorm2d(outC, momentum=momentum),
                                nn.ReLU())
    def forward(self, x):
        x = self.cT(x)
        return x

class double_conv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(double_conv, self).__init__()
        self.conv2x = nn.Sequential(
            conv(inC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum),
            conv(outC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum))   
    def forward(self, x):
        x = self.conv2x(x)
        return x

class inconv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(inconv, self).__init__()
        self.conv = double_conv(inC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum)
    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, inC, outC, momentum):
        super(down, self).__init__()
        self.go_down = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(inC, outC, kernel_size=3, padding=1, momentum=momentum))   
    def forward(self, x):
        x = self.go_down(x)
        return x

class up(nn.Module):
    def __init__(self, inC, outC, momentum):
        super(up, self).__init__()
        self.convt1 = convT(inC, outC, kernel_size=2, padding=0, stride=2, momentum=momentum)
        self.conv2x = double_conv(inC, outC, kernel_size=3, padding=1, momentum=momentum)
    
    def forward(self, x1, x2):
        x2 = self.convt1(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2x(x)
        return x

class outconv(nn.Module):
    def __init__(self, inC, outC, kernel_size, padding, momentum):
        super(outconv, self).__init__()
        self.conv = conv(inC, outC, kernel_size=kernel_size, padding=padding, momentum=momentum) 
    def forward(self, x):
        x = self.conv(x)
        return x
class ModelModule(pl.LightningModule):
    def __init__(self,opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt_Network=opt.Network
        self.opt_Loss=opt.Loss
        self.opt_Optim=opt.Optim
        self.set_Network()
        self.set_Loss()
        self.set_Optim()
        self.set_log_img_dir()
        self.h_weight=torch.tensor(scipy.io.loadmat(self.opt_Network.rpgd.h)['w']).float().cuda()
    def h(self,x):
        return nn.functional.conv2d(x, self.h_weight)
    def ht(self,y):
        return nn.functional.conv_transpose2d(y, self.h_weight)
    def set_Network(self,momentum=0.5):
        self.inc = inconv(1, 16, 3, 1, momentum)
        self.down1 = down(16, 32, momentum)
        self.down2 = down(32, 64, momentum)
        self.up1 = up(64, 32, momentum)
        self.up2 = up(32, 16, momentum)
        self.outc = outconv(16, 1, 3, 1, momentum)
    def forward(self, x):
        x_input = x.clone().detach()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x2, x3)
        x = self.up2(x1, x)
        x = self.outc(x)
        return x+x_input
    def set_Loss(self):
        self.Loss=nn.MSELoss()
    def set_Optim(self):
        self.Optim =  torch.optim.Adam(self.parameters(), **self.opt_Optim)
    def set_log_img_dir(self):
        self.log_val_img_dir=opj(os.getcwd(),'log_img','val')
        self.log_test_img_dir=opj(os.getcwd(),'log_img','test')
        os.makedirs(self.log_val_img_dir,exist_ok=True)
        os.makedirs(self.log_test_img_dir,exist_ok=True)
    def log_img(self,x0,xstar,xk,batch_idx,datadir):
        batch_idx=str(batch_idx)
        img_x0=x0.squeeze().detach().cpu().clip_(0,1).numpy().__mul__(255).astype(np.uint8)
        img_xstar=xstar.squeeze().detach().cpu().clip_(0,1).numpy().__mul__(255).astype(np.uint8)
        img_xk=xk.squeeze().detach().cpu().clip_(0,1).numpy().__mul__(255).astype(np.uint8)
        cv2.imencode('.png', img_x0)[1].tofile(opj(datadir,batch_idx+'_x0.jpg'))
        cv2.imencode('.png', img_xstar)[1].tofile(opj(datadir,batch_idx+'_xstar.jpg'))
        cv2.imencode('.png', img_xk)[1].tofile(opj(datadir,batch_idx+'_xk.jpg'))
    def training_step(self, batch, batch_idx):
        x1=batch['img_x1']
        y=batch['img_y']
        y_hat = self.forward(x1)
        loss = self.Loss(y_hat, y)
        if self.current_epoch>=50:
            x2=y_hat.clone().detach()
            y2_hat = self.forward(x2)
            loss+=self.Loss(y2_hat, y)
        if self.current_epoch>=100:
            x3=y.clone().detach()
            y3_hat = self.forward(x3)
            loss+=self.Loss(y3_hat, y)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        x1=batch['img_x1']
        y=batch['img_y']
        y_hat = self.forward(x1)
        loss = self.Loss(y_hat, y)
    def test(self, x0, t,gamma):
        c = self.opt_Network.rpgd.c
        alpha = self.opt_Network.rpgd.alpha
        x = x0.clone().detach()
        y = self.h(x0)
        best_snr = 0
        for k in range(self.opt_Network.rpgd.n_test_iter):
            z = self.forward(x)
            if k>0:
                if (z-x).pow(2).sum()>c*(z_prev-x_prev).pow(2).sum():
                    alpha = c*(z_prev-x_prev).pow(2).sum()/(z-x).pow(2).sum()*alpha
            x = alpha*z + (1-alpha)*x
            x = x - gamma*self.ht(self.h(x)-y)    
            loss = (self.h(x)-y).pow(2).sum()
            # self.log_img(x0,t,x,batch_idx,str(k),self.log_test_img_dir)
            snr, rec = RSNR(x ,t)
            if snr>best_snr:
                best_snr, best_rec = snr, rec  
            z_prev = z
            x_prev = x
            if k%self.opt_Network.rpgd.dk==self.opt_Network.rpgd.dk-1: 
                gamma /= self.opt_Network.rpgd.dgamma
        return best_snr, best_rec
    def test_step(self,batch, batch_idx):
        x0=batch['img_x1']
        t=batch['img_y']
        l = self.opt_Network.rpgd.gamma0[0]
        r = self.opt_Network.rpgd.gamma0[-1]
        snr_plot = np.array([])
        gamma = np.array([])
        gamma_tmp = self.opt_Network.rpgd.gamma0
        while abs(r-l)>self.opt_Network.rpgd.tol:
            best_snr = 0
            gamma = np.append(gamma, gamma_tmp)
            for i in gamma_tmp:
                snr, rec = self.test(x0=x0, t=t, gamma=i)
                if snr>best_snr: 
                    best_snr, best_rec = snr, rec
                    best_gamma = i
                snr_plot = np.append(snr_plot, snr)
            d = (r-l)/5
            l = best_gamma -d
            r = best_gamma +d
            gamma_tmp = np.linspace(l, r, 4)
        arg = np.argsort(gamma)
        self.log_img(x0,t,best_rec,batch_idx,self.log_test_img_dir)
        print(batch_idx,best_snr)
        return best_snr, best_rec
    def configure_optimizers(self):
        scheduler = torch.optim.lr_scheduler.CyclicLR(self.Optim, self.opt_Optim.lr, 1e-3, step_size_up=500, step_size_down=500,cycle_momentum=False)
        return [self.Optim],[scheduler]
def RSNR(rec,target):
    with torch.no_grad():
        rec = compress3d(rec)
        target = compress3d(target)
        
        rec_size = rec.size()
        rec = rec.cpu().numpy()
        target = target.cpu().numpy()
        rec=rec.reshape(np.size(rec))
        target=target.reshape(np.size(rec))
        slope, intercept, _, _, _ = stats.linregress(rec,target)
        rec=slope*rec+intercept

        return 10*np.log10(sum(target**2)/sum((rec-target)**2)), torch.Tensor(rec).view(rec_size).cuda()
    
def compress2d(x):
    '''compress [1,c,h,w] to [h,w] by root-mean-square at dimension 1, without keeping the dimension, then squeeze dimension 0 '''
    if len(x.size())>2:
        return torch.sqrt(x.pow(2).sum(dim=1))[0]
    return x
def compress3d(x):
    '''compress [1,c,h,w]-->[1,h,w] by root-mean-square at dimension 1, without keeping the dimension'''
    if len(x.size())>3:
        return torch.sqrt(x.pow(2).sum(dim=1))
    return x
def compress4d(x):
    '''compress [1,c,h,w]-->[1,1,h,w] by root-mean-square at dimension 1 and keep the dimension'''
    return torch.sqrt(x.pow(2).sum(dim=1, keepdim=True))
