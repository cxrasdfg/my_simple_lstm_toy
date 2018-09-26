import torch as th 

class LSTMLayer(th.nn.Module):
    def __init__(self,hidden_units,voc_size):
        super(LSTMLayer,self).__init__()
        
        self.W_f=th.nn.Parameter(th.randn((voc_size+hidden_units),hidden_units).t())
        self.b_f=th.nn.Parameter(th.randn(hidden_units))
        
        self.W_i=th.nn.Parameter(th.randn((voc_size+hidden_units),hidden_units).t())
        self.b_i=th.nn.Parameter(th.randn(hidden_units))
        
        self.W_c=th.nn.Parameter(th.randn((voc_size+hidden_units),hidden_units).t())
        self.b_c=th.nn.Parameter(th.randn(hidden_units))
        
        self.W_o=th.nn.Parameter(th.randn((voc_size+hidden_units),hidden_units).t())
        self.b_o=th.nn.Parameter(th.randn(hidden_units))
        
        self.W_v=th.nn.Parameter(th.randn((hidden_units,voc_size)).t())
        self.b_v=th.nn.Parameter(th.randn(voc_size))

        self.hidden_units=hidden_units
        self.voc_size=voc_size

    def forward(self,x_t,h_t_,c_t_):
        r"""
        Args:
            x_t_ (tensor[float32]): [m,b]
            h_t_ (tensor[float32]): [h,b]
            c_t_ (tensor[float32]): [h,b]
        Return:
            out_t(tensor[float32]): [m,b]
            h_t (tensor[float32]): [h,b]
            c_t (tensor[float32]): [h,b]
        """
        _,b=x_t.shape

        # forward...
        # x_t:[voc_size,b],y_t:[voc_size,b]
        x_t=th.cat([h_t_,x_t],dim=0) # [hidden_units+voc_size,b]
        
        # first step
        f_t=self.W_f.mm(x_t)+self.b_f[:,None] # [hidden_units,b]
        f_t=f_t.sigmoid()
        
        # second_step
        i_t=self.W_i.mm(x_t)+self.b_i[:,None] # [hidden_units,b]
        i_t=i_t.sigmoid()

        C_t_=self.W_c.mm(x_t)+self.b_c[:,None] # [hidden_units,b]
        C_t_=C_t_.tanh()

        C_t=f_t * c_t_+i_t *C_t_ # [hidden_units,b]
        
        # third step
        o_t=self.W_o.mm(x_t)+self.b_o[:,None] # [hidden_units,b]
        o_t=o_t.sigmoid()
        h_t=o_t*C_t.tanh() # [hidden_units,b]
        
        # output
        out_=self.W_v.mm(h_t)+self.b_v[:,None] # [voc_size,b]
        out_t=th.nn.functional.softmax(out_,dim=0)
        
        
        return out_t,h_t,C_t