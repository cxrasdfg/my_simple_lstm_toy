# coding=utf-8
import torch as th 
from config import cfg

class LSTM(th.nn.Module):
    def __init__(self,hidden_units,voc_size):
        r"""Constructor
        Args:
            hidden_units (int): the number of units in hiden layers...
            voc_size (int): size of the vocabulary
        """
        super(LSTM,self).__init__()

        # seems it could be euqal...
        hidden_units=voc_size
        # prepare the parameters
        self.W_f=th.nn.Parameter(th.randn((voc_size+hidden_units),hidden_units) ).t()
        self.b_f=th.nn.Parameter(th.randn(hidden_units))
        
        self.W_i=th.nn.Parameter(th.randn((voc_size+hidden_units),hidden_units)).t()
        self.b_i=th.nn.Parameter(th.randn(hidden_units))
        
        self.W_c=th.nn.Parameter(th.randn((voc_size+hidden_units),hidden_units)).t()
        self.b_c=th.nn.Parameter(th.randn(hidden_units))
        
        self.W_o=th.nn.Parameter(th.randn((voc_size+hidden_units),hidden_units)).t()
        self.b_o=th.nn.Parameter(th.rannd(hidden_units))

        self.hidden_units=hidden_units
        self.voc_size=self.voc_size

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
            b,_,_=X.shape

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
        for x_t,y_t in zip(X,Y):
            # x_t:[voc_size,b],y_t:[voc_size,b]
            x_t=th.cat([hidden_res[-1],x_t],dim=0) # [hidden_units+voc_size,b]
            
            # first step
            f_t=self.W_f.mm(x_t)+self.b_f[:,None] # [hidden_units,b]
            f_t=f_t.sigmoid()
            
            # second_step
            i_t=self.W_i.mm(x_t)+self.b_i[:,None] # [hidden_units,b]
            i_t=i_t.sigmoid()

            C_t_=self.W_c.mm(x_t)+self.b_c[:,None] # [hidden_units,b]
            C_t_=C_t_.tanh()

            C_t=f_t * cell_res[-1]+i_t *C_t_ # [hidden_units,b]
            
            # third step
            o_t=self.W_o.mm(x_t)+self.b_o[:,None] # [hidden_units,b]
            o_t=o_t.sigmoid()
            h_t=o_t*C_t.tanh() # [hidden_units,b]=[voc_size,b]

            hidden_res.append(h_t)
            cell_res.append(C_t)
            final_output.append(h_t.softmax(dim=0))  
        
        final_output = th.cat([o[None] for o in final_output],dim=0) # [seq_len,voc_size,b]
        if self.training:
            # prepare loss function
            # Y: [seq_len,voc_size,b]
            loss=-final_output[Y>0].log().sum()

            return loss/b
        else:
            return final_output.permute(2,0,1).contiguous() # [b,seq_len,voc_size]
        