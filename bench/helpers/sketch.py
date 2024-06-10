import numpy as np

import torch
import torch.nn as nn

def topk(vec, k):
    """ Return the largest k elements (by magnitude) of vec"""
    ret = torch.zeros_like(vec)

    # on a gpu, sorting is faster than pytorch's topk method
    topkIndices = torch.sort(vec**2)[1][-k:]
    #_, topkIndices = torch.topk(vec**2, k)

    ret[topkIndices] = vec[topkIndices]
    return ret

def printMemoryUsage():
    import gc
    bigs = []
    totalBytes = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
                if isinstance(obj, torch.cuda.ByteTensor):
                    dsize = 1
                elif isinstance(obj, torch.cuda.FloatTensor) or isinstance(obj, torch.cuda.IntTensor):
                    dsize = 4
                elif isinstance(obj, torch.cuda.DoubleTensor) or isinstance(obj, torch.cuda.LongTensor):
                    dsize = 8
                totalBytes += np.product(obj.size()) * dsize
                if obj.size()[0] > 90000000:
                    bigs.append(obj)
        except:
            pass
    for big in bigs:
        print(big)
    print("Total Size: {} MB".format(totalBytes / 1024 / 1024))

    