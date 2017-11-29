from model import NetD, NetG
import torch
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

nz = 100
cl = 10
bs = 1
transform = transforms.Compose([transforms.ToPILImage(),])
labels_list = [np.random.randint(0,10)]
noise = Variable(torch.rand(bs, nz))
onehot_labels = np.zeros((bs, cl))
onehot_labels[np.arange(bs), labels_list] = 1
onehot_labels = Variable(torch.from_numpy(onehot_labels).type(torch.FloatTensor)) 

noise = torch.cat([noise, onehot_labels], 1)
noise = noise.view(bs, nz+cl, 1,1)

netG = NetG(nz+cl)
netG.load_state_dict(torch.load('checkpoints/netG__epoch_25.pth', map_location=lambda storage, loc: storage))

output = netG(noise)
imgs = transform(output[0].data)
print('class: ' + str(labels_list[0]))
imgs.show()
