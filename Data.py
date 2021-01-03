import pytorch_lightning as pl
import scipy.io
from torch.utils.data import DataLoader
import numpy as np
import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self,opt):
        self.opt=opt
        self.input_path = opt.input_path
        self.target_path = opt.target_path
        self.length = opt.length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        # 1. Read data
        input_loadpath = self.input_path + str(idx+1) +'.mat'
        input_image = torch.tensor(scipy.io.loadmat(input_loadpath)['x'])
        target_loadpath = self.target_path + str(idx+1) + '.mat'
        target_image = torch.tensor(scipy.io.loadmat(target_loadpath)['x'])
        input_image = input_image.float()
        target_image = target_image.float()
        if len(input_image.size())==2:
            input_image.unsqueeze_(0)
        if len(target_image.size())==2:         
            target_image.unsqueeze_(0)
        return {'img_x1': input_image, 'img_y': target_image,}
    
def collate_fn(data_list):
    img_x1 = torch.stack([data['img_x1'] for data in data_list ], dim=0)
    img_y = torch.stack([data['img_y'] for data in data_list], dim=0)
    batch = {'img_x1': img_x1, 'img_y':img_y,}
    return batch

class DataModule(pl.LightningDataModule):

    def __init__(self,opt):
        super().__init__()
        self.opt_Dataset=opt.Dataset
        self.opt_Dataloader=opt.Dataloader
    def setup(self,stage=None):
        # Dataset
        self.Dataset_train=Dataset(self.opt_Dataset.train)
        self.Dataset_val=Dataset(self.opt_Dataset.val)
        self.Dataset_test=Dataset(self.opt_Dataset.test)
    def train_dataloader(self):
        return DataLoader(self.Dataset_train,collate_fn=collate_fn,**self.opt_Dataloader.train)

    def val_dataloader(self):
        return DataLoader(self.Dataset_val,collate_fn=collate_fn,**self.opt_Dataloader.val)

    def test_dataloader(self):
        return DataLoader(self.Dataset_test,collate_fn=collate_fn,**self.opt_Dataloader.test)