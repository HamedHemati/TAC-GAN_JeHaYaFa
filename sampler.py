import torch
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
from model import NetG
from data_loader import ImTextDataset
from torch.utils.data import DataLoader


data_root = 'Datasets/Oxford Flowers Dataset'
imText_ds = ImTextDataset(data_root,dataset='flowers', train=False)
dl = DataLoader(imText_ds, shuffle=True, batch_size=1)

trans_toPIL = transforms.ToPILImage() 
noise = Variable(torch.randn(1,100))
skv = Variable(torch.randn(1,4800))
path = 'checkpoints/netG__epoch_60.pth'
netG = NetG()
netG.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

for i, (img,lbl,cap) in enumerate(dl):
    if i%10==4:
        cap = Variable(cap)
        out = netG(noise,cap)
        img = trans_toPIL(out.data[0])
        img.show()
        break