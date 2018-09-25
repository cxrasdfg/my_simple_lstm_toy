# coding=utf-8

import sys
from config import cfg

def train():
    cfg._print()

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