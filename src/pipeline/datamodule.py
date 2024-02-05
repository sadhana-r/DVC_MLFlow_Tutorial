
import torch
from torch.utils.data import random_split
import lightning as L
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from monai.data import DataLoader, Dataset
from monai.transforms import ( Compose, Resized, 
                              ToTensord, NormalizeIntensityd, 
                              EnsureChannelFirstd, 
                              RandAffined )
import numpy as np
import SimpleITK as sitk
import pandas as pd


# Only applied to training dataset
class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample

    """
    def __init__(self, dataset, transform=None):

        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.dataset)
    
# Takes as input the path to files
class LungSegmentationDataset(Dataset):
    def __init__(self, datalist_filepath, transforms):
        
        self.data_list = pd.read_csv(datalist_filepath)
        self.transform = transforms

    def __len__(self):
        return self.data_list.shape[0]
    
    def file_path_to_ID(self,filepath):
        id = os.path.basename(filepath)[:-6]
        return id
    
    def __getitem__(self, idx):

        # Load Image and corresponding segmentation masks
        img_name = self.data_list.loc[idx,'image']
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))

        # Shenzen images have 3 channels but all of them are the same
        if len(img.shape) == 3:
            img = img[:,:,0]

        if self.data_list.loc[idx,'source'] == 'montgomery':
            left_name = self.data_list.loc[idx,'seg_left']
            right_name = self.data_list.loc[idx,'seg_right']
            
            left = sitk.GetArrayFromImage(sitk.ReadImage(left_name))/255
            right = sitk.GetArrayFromImage(sitk.ReadImage(right_name))/255

            seg = left + right #  Combine the right and left masks

        elif self.data_list.loc[idx,'source'] == 'shenzen':
            seg_name = self.data_list.loc[idx,'seg']
            seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_name))/255

        seg = seg.astype(int)
        sample = {'image':img,'label':seg,'id': self.file_path_to_ID(img_name)}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
    

class LungSegmentationDataModule(L.LightningDataModule):
    def __init__(self, datalist_file, batch_size, input_size, num_workers):
            super().__init__()
            self.datalist_file = datalist_file
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.input_size = input_size

            self.dims = 2

            self.transforms = Compose([EnsureChannelFirstd(keys=["image","label"], channel_dim='no_channel'), 
                Resized(keys=['image','label'], spatial_size = self.input_size, mode=("bilinear","nearest")),
                NormalizeIntensityd(keys=['image']),
                ToTensord(keys=["image","label"])])
                
            self.aug_transforms = RandAffined(keys=['image','label'],mode=('bilinear','nearest'),prob=0.1,rotate_range=(0, 0, np.pi), padding_mode="zeros")

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        This function is used if downloading data . Processes that occcur on a single processing unit
        """
        pass     

    def setup(self, stage):

        # Assign train/val datasets for use in datlaloaders
        entire_dataset =  LungSegmentationDataset(
            self.datalist_file,
            transforms = self.transforms)
            
        # Split data
        generator = torch.Generator().manual_seed(42) # reproducible split
        self.train_ds, self.val_ds, self.test_ds = random_split(entire_dataset,[0.7,0.2,0.1], generator=generator)

        # Apply data augmentation transfprms to train_ds
        self.train_ds = ApplyTransform(self.train_ds, self.aug_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle=True)
            
    def val_dataloader(self):
        return DataLoader(self.val_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle=False)           