import torch
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
from model import NetG
import pickle


# convert to PIL Image
trans_toPIL = transforms.ToPILImage()

# load the model
checkpoint_path = 'checkpoints/netG__epoch_100.pth'
n_l = 150
n_z = 100
n_c = 128
netG = NetG(n_z=n_z, n_l=n_l, n_c=n_c)
netG.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def generate_from_caption():
    caption_file = "enc_text.pkl"
    # load encoded captions
    train_ids = pickle.load(open(caption_file, 'rb'))
    num_captions = len(train_ids['features'])
    num_images = 2

    # create random noise


    #create random caption
    skv = Variable(torch.randn(num_images,4800))
    skv.data.normal_(0,1.1)


    for i in range(num_captions):
        noise = Variable(torch.randn(num_images,n_z))
        noise.data.normal_(0,2)
        caption = Variable(torch.from_numpy(train_ids['features'][i]))

        for j in range(num_images):
            out = netG(noise[j].view(1,noise.size(1)),caption.view(1,caption.size(0)))
            img = trans_toPIL(out.data[0])
            img.save(str(i)+str(j)+'.png')

# other fun experiments

def interpolate(inb=5):
    cap1 = Variable(torch.randn(1,4800))
    cap2 = Variable(torch.randn(1,4800))
    cap1.data.normal_(0,5)
    cap2.data.normal_(0,5)

    for i in range(inb):
        alpha = i/float(inb)
        cap = alpha*cap1 + (1-alpha)*cap2
        noise = Variable(torch.rand(1,n_z))
        noise.data.normal_(0,1)
        out = netG(noise,cap)
        img = trans_toPIL(out.data[0])
        img.save('interp'+str(i)+'.png')


def addDiff():
    cap1 = Variable(torch.randn(1,4800))
    cap2 = Variable(torch.randn(1,4800))
    cap3 = Variable(torch.randn(1,4800))
    cap1.data.normal_(0,5)
    cap2.data.normal_(0,5)
    cap3.data.normal_(0,5)

    diff = cap1-cap2
    final = cap3+diff
    noise = Variable(torch.rand(1,100))
    noise.data.normal_(0,1)

    out = netG(noise,cap1)
    img = trans_toPIL(out.data[0])
    img.save('im1.png')

    out = netG(noise,cap2)
    img = trans_toPIL(out.data[0])
    img.save('im2.png')

    out = netG(noise,cap3)
    img = trans_toPIL(out.data[0])
    img.save('im3.png')

    out = netG(noise,final)
    img = trans_toPIL(out.data[0])
    img.save('final.png')

    out = netG(noise,diff)
    img = trans_toPIL(out.data[0])
    img.save('diff.png')
