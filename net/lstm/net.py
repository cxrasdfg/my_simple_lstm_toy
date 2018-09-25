# coding=utf-8
import torch as th 

class LSTM(th.nn.Module):
    def __init__(self,hidden_units,voc_size):
        r"""Constructor
        Args:
            hidden_units (int): the number of units in hiden layers...
            voc_size (int): size of the vocabulary
        """
        super(LSTM,self).__init__()
        self.W_f=th.nn.Parameter(torch.randn((voc_size+hidden_units),voc_size) )
        self.b_f=th.nn.Parameter(torch.randn(voc_size))

    def forward(self,*args):
        if self.training:
            X,Y=args
            
        else:
            raise NotImplementedError()