import pytorch_lightning as pl
import hydra
import os
from os.path import join as opj
from os.path import dirname as opd

from Data import DataModule
from Model import ModelModule
@hydra.main(config_path=opj(opd(__file__),'opt'),config_name='train')
def main(opt):
    pl.seed_everything(1)
    data=DataModule(opt.Data)
    model = ModelModule(opt.Model)
    trainer = pl.Trainer(**opt.trainer)
    trainer.fit(model,data)
    trainer.test(datamodule=data)
if __name__ == '__main__':
    main()