import torch
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
from model import NetG
from data_loader import ImTextDataset
from torch.utils.data import DataLoader

'''
This is just a random sampler: random noise+random encoded caption
'''

#data_root = 'Datasets/Oxford Flowers Dataset'
#imText_ds = ImTextDataset(data_root,dataset='flowers', train=False)
#dl = DataLoader(imText_ds, shuffle=True, batch_size=1)

trans_toPIL = transforms.ToPILImage() 
noise = Variable(torch.randn(1,100))
skv = Variable(torch.randn(1,4800))
path = 'checkpoints/netG__epoch_100.pth'
netG = NetG()
netG.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

out = netG(noise,skv)
img = trans_toPIL(out.data[0])
img.show()
