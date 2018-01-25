import math
import time
import torch
from torch.autograd import Variable

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def create_var(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    return Variable(tensor)

def cuda_model(model):
    if torch.cuda.is_available():
        return model.cuda()
    return model