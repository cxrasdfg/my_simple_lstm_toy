# coding=utf-8
import torch as th 
from config import cfg
from collections import OrderedDict
from .layers import LSTMLayer

class LSTM(th.nn.Module):
    def __init__(self,hidden_units,voc_size,num_layers):
        r"""Constructor
        Args:
            hidden_units (int): the number of units in hiden layers...
            voc_size (int): size of the vocabulary
        """
        super(LSTM,self).__init__()

        layers=[]
        for i in range(num_layers):
            layers.append(['%d-lstm'%i,LSTMLayer(hidden_units,voc_size)])

        self.layers=th.nn.Sequential(OrderedDict(layers))
        self.num_lstm= num_layers
        self.hidden_units=hidden_units
        self.voc_size=voc_size

        self.get_optimizer()
    
    def get_optimizer(self,lr=cfg.lr,use_adam=cfg.use_adam,weight_decay=cfg.weight_decay):
        params=[]
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        if use_adam:
            print("Using Adam optimizer")
            self.optimizer = th.optim.Adam(params)
        else:
            print("Using SGD optimizer")
            self.optimizer = th.optim.SGD(params, momentum=0.9)
        return self.optimizer
    
    def opt_step(self,loss):
        r"""Use the loss to backward the net, 
        then the optimizer will update the weights
        which requires grad...
        Args:
            loss (tensor(float32)): loss 
        Return:
            loss (tensor[float32]): value of current loss...
        """
        
        # grad descend
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.parameters(),10)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def forward(self,*args):
        if self.training:
            X,Y=args # [b,seq_len,voc_size] 
            b,seq_len,_=X.shape

            X,Y=X.permute(1,2,0).contiguous(),Y.permute(1,2,0).contiguous() # [seq_len,voc_size,b]
        else:
            X,=args # [b,seq_len,voc_size]
            b,_,_=X.shape
            X=X.permute(1,2,0).contiguous() # [seq_len,voc_size,b]

        # store the res in hidden layers
        hidden_res=[]
        
        # store the res of cell
        cell_res=[]
        
        # store res of the final output (after softmax)
        final_output=[]
        
        # extern_input is for the first position char in the sentences since it has no pre-hidden input
        extern_input=th.ones([self.hidden_units,1 ]).type_as(X)# [hidden_units,1]
        hidden_res.append(extern_input.expand(-1,b)) 
        cell_res.append(hidden_res[-1].detach())
        
        # forward...
        for x_t in X:
            # x_t:[voc_size,b],y_t:[voc_size,b]
            
            temp=th.full([self.num_lstm,self.voc_size,b],1).type_as(x_t)
            temp[0]=x_t

            h_t_=hidden_res[-1]
            c_t_=cell_res[-1]
            for (_,m),x_t_ in zip(self.layers.named_children(),temp):
                out_t_,h_t_,c_t_=m(x_t_,h_t_,c_t_)
            
            hidden_res.append(h_t_)
            cell_res.append(c_t_)
            final_output.append(out_t_)  
        
        final_output = th.cat([o[None] for o in final_output],dim=0) # [seq_len,voc_size,b]
        if self.training:
            # prepare loss function
            # Y: [seq_len,voc_size,b]
            loss=-final_output[Y>0].log().sum()
            loss/=seq_len

            return loss/b
        else:
            return final_output.permute(2,0,1).contiguous() # [b,seq_len,voc_size]
        