# coding=utf-8

import sys
from config import cfg
from data import TextDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def train():
    cfg._print()
    data_set=TextDataset(cfg.seq_len,cfg.batch_size,cfg.txt_file_path)
    data_loader=DataLoader(data_set,batch_size=cfg.batch_size,shuffle=True,
    drop_last=False)

    for input_,target_ in tqdm(data_loader):
        # print(input_.shape,target_.shape)
        pass
    

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