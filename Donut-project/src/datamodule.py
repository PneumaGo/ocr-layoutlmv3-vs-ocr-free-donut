import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from pathlib import Path

class SROIEDataModule(pl.LightningDataModule):
    def __init__(self, train_img_dir, train_ent_dir, test_img_dir, test_ent_dir, processor, batch_size=1):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.train_ent_dir = train_ent_dir
        self.test_img_dir = test_img_dir
        self.test_ent_dir = test_ent_dir
        
        self.processor = processor
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = DonutSROIEDataset(
            self.train_img_dir, 
            self.train_ent_dir, 
            self.processor, 
            split="train"
        )
        
        self.val_ds = DonutSROIEDataset(
            self.test_img_dir, 
            self.test_ent_dir, 
            self.processor, 
            split="val"
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )