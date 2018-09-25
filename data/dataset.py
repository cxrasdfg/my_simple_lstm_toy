# coding=utf-8


import torch as th 
from torch.utils.data import Dataset
from config import cfg

class TextDataset(Dataset):
    def __init__(self,seq_len,batch_size,txt_path):
        assert txt_path is not None

        self.seq_len=seq_len
        self.batch_size=batch_size

        content=open(txt_path,'rb').read().decode('utf-8')
        
        # build vocabulary...
        self.vocabulary=set(content)
        # use `i+1` to preserve the 0-th location for unknown
        self.vocabulary,self.vocabulary_for_decoding\
            ={c:i+1 for i,c in enumerate(self.vocabulary)},\
            {i+1:c for i,c in enumerate(self.vocabulary)}
        
        # add `0`
        self.vocabulary_for_decoding[0]='unkonwn'
        self.content=content # [size]

        n_batches=len(self.content)//(seq_len*batch_size)
        
        self.data_size=n_batches*batch_size

        # drop for division
        self.content=self.content[:n_batches*seq_len*batch_size]

    def encode(self,input_str):
        r"""Encode the input string to an one-hot vector
        Args:
            input_str (str): [n]
        Return:
            encoded_str (tensor[float32]): [n,m], m is the size of the vocabulary
            targets (tensor[float32]): [n,m]
        """
        n=len(input_str)
        m=len(self.vocabulary)
        encoded_str=th.full([n,m+1],0) # [n,m]
        encoded_id=[self.vocabulary[c] for c in input_str] # [n]
        encoded_str[ [_ for _ in range(n)], encoded_id]=1. # [n,m]

        targets=encoded_str.detach().clone()
        targets[:,:-1],targets[-1]=\
            targets[:,1:].clone(),targets[0].clone()
        
        return encoded_str,targets
    
    def decode(self,encoded_str):
        r"""Decode the output to normal string
        Args:
            encoded_str (tensor[float32]): [n,m]
        Return:
            decoded_str (str): [n]
        """
        _,pred_labels=encoded_str.max(dim=1)
        decoded_str=''
        for pred_label in pred_labels:
            c=self.vocabulary_for_decoding[pred_label]
            decoded_str+=c
        return decoded_str

    def __getitem__(self,idx):
        sub_content=self.content[idx*self.seq_len:(idx+1)*self.seq_len] # [seq_len]
        encoded_sub_content,targets=self.encode(sub_content) # [seq_len,m],[seq_len,m]
        
        return encoded_sub_content,targets

    def __len__(self):
        
        return self.data_size