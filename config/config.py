# coding=utf-8

class CFG():
    txt_file_path='./data/sources/anna.txt'
    device_id=0
    use_cuda=True

    weights_dir='./weights/'
    rand_seed=1234567
    
    seq_len=100
    batch_size=128
    num_worker=8
    
    epochs=64
    save_per_epoch=1
    eval_per_epoch=1

    lr=1e-5
    lrs={'40000':lr,'50000':lr/10.,'60000':lr/100.}
    
    weight_decay=0.0005
    use_adam=False

    def _print(self):
        print('Config')
        print('{')
        for k in self._attr_list():
            print('%s=%s'% (k,getattr(self,k)) )
        print("}")
    @staticmethod

    def _attr_list():
        return [k for k in CFG.__dict__.keys() if not k.startswith('_') ] 
            
    # rpn path
   
cfg=CFG()