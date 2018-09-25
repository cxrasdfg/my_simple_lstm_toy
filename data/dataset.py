# coding=utf-8


import torch as th 
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self,txt_path):
        assert txt_path is not None

        raise NotImplementedError('')
    
    def __getitem__(self,idx):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()