# coding=utf-8

import sys
from config import cfg
import torch as th
import numpy as np

np.random.seed(cfg.rand_seed)
th.manual_seed(cfg.rand_seed)
th.cuda.manual_seed(cfg.rand_seed)

from data import TextDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import re,os
from net import LSTM
from tensorboardX import SummaryWriter

def get_check_point():
    pat=re.compile("""weights_([\d]+)_([\d]+)""")
    base_dir=cfg.weights_dir
    w_files=os.listdir(base_dir)
    if len(w_files)==0:
        return 0,0,None
    w_files=sorted(w_files,key=lambda elm:int(pat.match(elm)[1]),reverse=True)

    w=w_files[0]
    res=pat.match(w)
    epoch=int(res[1])
    iteration=int(res[2])

    return epoch,iteration,base_dir+w


def adjust_lr(opt,iters,lrs=cfg.lrs):
    lr=0
    for k,v in lrs.items():
        lr=v
        if iters<int(k):
            break

    for param_group in opt.param_groups:
        
        param_group['lr'] = lr


def train():
    cfg._print()
    writer = SummaryWriter('log')

    data_set=TextDataset(cfg.seq_len,cfg.batch_size,cfg.txt_file_path)

    data_loader=DataLoader(
        data_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_worker
    )
    
    # NOTE: plus one, en?
    net=LSTM(300,len(data_set.vocabulary))
    # net._print()
    epoch,iteration,w_path=get_check_point()
    if w_path:
        model=th.load(w_path)
        net.load_state_dict(model)
        print("Using the model from the last check point:%s"%(w_path) )
        epoch+=1

    net.train()
    is_cuda=cfg.use_cuda
    did=cfg.device_id
    if is_cuda:
        net.cuda(did)

    while epoch<cfg.epochs:
        
        # print('********\t EPOCH %d \t********' % (epoch))
        for X,Y in tqdm(data_loader):
            if is_cuda:
                X=X.cuda(did)
                Y=Y.cuda(did)

            _loss=net(X,Y)
            _loss=net.opt_step(_loss)
            writer.add_scalar('Train/Loss',_loss,iteration)
            tqdm.write('Epoch:%d, iter:%d, loss:%.5f'%(epoch,iteration,_loss))

            iteration+=1
            adjust_lr(net.optimizer,iteration,cfg.lrs)

        if epoch % cfg.save_per_epoch ==0:
            th.save(net.state_dict(),'%sweights_%d_%d'%(cfg.weights_dir,epoch,iteration) )
            
        epoch+=1

    # print(eval_net(net=net))

    

def eval():
    raise NotImplementedError()

def test():
    raise NotImplementedError()

if __name__ == '__main__':
    print('simple lstm toy example...')
    if len(sys.argv) == 1:
        opt='train'
    else:
        opt=sys.argv[1]

    if opt == 'train':
        train()
    elif opt== 'test':
        test()
    elif opt=='eval':
        eval()
    else:
        assert 0,'args must be `train`, `test` or `eval`'