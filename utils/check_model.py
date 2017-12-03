import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
sys.path.append('..')
from model import NetD, NetG


# validate NetD
netD = NetD()
b_size = 5
img_size = 128
images = Variable(torch.rand(b_size, 3, img_size, img_size))
skip_v = Variable(torch.randn(b_size, 4800))

o1, o2 = netD(images, skip_v)
print(o1.size())
print(o2.size())
print("NetD works fine")


# validate NetG
netG = NetG()
b_size = 5
n_size = 100
noise = Variable(torch.randn(b_size, 100))
skip_v = Variable(torch.randn(b_size, 4800))
o = netG(noise, skip_v)
print(o.size())
print("NetG  works fine")