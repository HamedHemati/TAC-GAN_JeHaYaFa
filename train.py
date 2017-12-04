import argparse
import os
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import  Dataset, DataLoader
from model import NetD, NetG
import torchvision.datasets as dset
import torchvision.transforms as transforms
from data_loader import ImTextDataset


class TACGAN():

    def __init__(self, args):
        self.lr = args.lr
        self.cuda = args.use_cuda
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.epochs = args.epochs
        self.data_root = args.data_root
        self.dataset = args.dataset
        self.num_classes = args.num_cls
        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix
        self.continue_training = args.continue_training
        self.netG_path = args.netg_path
        self.netD_path = args.netd_path
        self.save_after = args.save_after
        self.trainset_loader = None
        self.evalset_loader = None  
        self.num_workers = args.num_workers
        self.n_z = 100 # length of the noise vector
        self.bce_loss = nn.BCELoss()
        self.nll_loss = nn.NLLLoss()
        self.netD = NetD(n_cls=self.num_classes)
        self.netG = NetG(n_z=self.n_z)
        
        # convert to cuda tensors
        if self.cuda and torch.cuda.is_available():
            print('CUDA is enabled')
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.nll_loss = self.nll_loss.cuda() 

        # optimizers for netD and netG
        self.optimizerD = optim.Adam(params=self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizerG = optim.Adam(params=self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # create dir for saving checkpoints if does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    # start training process
    def train(self):
        # load trainset and evalset
        imtext_ds = ImTextDataset(data_dir=self.data_root, dataset=self.dataset, train=True, image_size=self.image_size)
        self.trainset_loader = DataLoader(dataset=imtext_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
        print("Dataset loaded successfuly")
        # load checkpoints for continuing training
        if args.continue_training:
            self.loadCheckpoints()
             
        # repeat for the number of epochs
        for epoch in range(self.epochs):
            self.trainEpoch(epoch)
            #self.evalEpoch(epoch)
            self.saveCheckpoints(epoch)

    # train epoch
    def trainEpoch(self, epoch):
        self.netD.train() # set to train mode
        self.netG.train() #! set to train mode???
    
        netd_loss_sum = 0
        netg_loss_sum = 0
        start_time = time()
        for i, (images, labels, captions) in enumerate(self.trainset_loader):
            batch_size = images.size(0) # !batch size my be different (from self.batch_size) for the last batch
            images, labels, captions = Variable(images), Variable(labels), Variable(captions) # !labels should be LongTensor
            lbl_real = Variable(torch.ones(batch_size, 1))
            lbl_fake = Variable(torch.zeros(batch_size, 1))
            noise = Variable(torch.randn(batch_size, self.n_z)) # create random noise
            rnd_perm = torch.randperm(batch_size)
            if self.cuda:
                images, labels, captions = images.cuda(), labels.cuda(), captions.cuda()
                lbl_real, lbl_fake = lbl_real.cuda(), lbl_fake.cuda()
                noise = noise.cuda()
                rnd_perm = rnd_perm.cuda()
            
            ############### Update NetD ###############
            self.netD.zero_grad()       
            # train with wrong image, wrong label, real caption
            outD_wrong, outC_wrong = self.netD(images[rnd_perm], captions)
            lossD_wrong = self.bce_loss(outD_wrong, lbl_fake)
            lossC_wrong = self.nll_loss(outC_wrong, torch.max(labels[rnd_perm],1)[1])

            # train with real image, real label, real caption
            outD_real, outC_real = self.netD(images, captions)
            lossD_real = self.bce_loss(outD_real, lbl_real)
            lossC_real = self.nll_loss(outC_real, torch.max(labels,1)[1])

            # train with fake image, real label, real caption
            fake = self.netG(noise, captions)
            outD_fake, outC_fake = self.netD(fake.detach(), captions)
            lossD_fake = self.bce_loss(outD_fake, lbl_fake)
            lossC_fake = self.nll_loss(outC_fake, torch.max(labels,1)[1])
            
            # backward and forwad for NetD
            netD_loss = lossC_wrong+lossC_real+lossC_fake + lossD_wrong+lossD_real+lossD_fake
            netD_loss.backward()
            self.optimizerD.step()      

            ########## Update NetG ##########
            # train with fake data
            self.netG.zero_grad()
            d_fake, c_fake = self.netD(fake, captions)
            lossD_fake_G = self.bce_loss(d_fake, lbl_real)
            lossC_fake_G = self.nll_loss(c_fake, torch.max(labels,1)[1])
            netG_loss = lossD_fake_G + lossC_fake_G 
            netG_loss.backward()
            self.optimizerG.step()
            
            netd_loss_sum += netD_loss.data[0]
            netg_loss_sum += netG_loss.data[0]
            ### print progress info ###
            print('Epoch %d/%d, %.2f%% completed. Loss_NetD: %.4f, Loss_NetG: %.4f'
                  %(epoch, self.epochs,(float(i)/len(self.trainset_loader))*100, netD_loss.data[0], netG_loss.data[0]))
       
        end_time = time()
        if i%5 == 0:
            epoch_time = (end_time-start_time)/60
            netd_avg_loss = netd_loss_sum / len(self.trainset_loader)
            netg_avg_loss = netg_loss_sum / len(self.trainset_loader)
            log_msg = '-------------------------------------------\n'
            log_msg += 'Epoch %d took %.2f minutes\n'%(epoch, epoch_time)
            log_msg += 'NetD average loss: %.4f, NetG average loss: %.4f\n' %(netd_avg_loss, netg_avg_loss)
            log_msg += '-------------------------------------------\n\n'
            print(log_msg)
            with open(os.path.join(self.save_dir, 'training_log'),'a') as log_file:
                log_file.write(log_msg)

    # eval epoch                   
    def evalEpoch(self, epoch):
        #self.netD.eval()
        #self.netG.eval()
        return 0

    # save after each epoch
    def saveCheckpoints(self, epoch):
        if epoch%self.save_after==0:
            name_netD = "netD_" + self.save_prefix + "_epoch_" + str(epoch) + ".pth"
            name_netG = "netG_" + self.save_prefix + "_epoch_" + str(epoch) + ".pth"
            torch.save(self.netD.state_dict(), os.path.join(self.save_dir, name_netD))
            torch.save(self.netG.state_dict(), os.path.join(self.save_dir, name_netG))
            print("Checkpoints for epoch %d saved successfuly" %(epoch))

    # load checkpoints to continue training
    def loadCheckpoints(self):
        self.netG.load_state_dict(torch.load(self.netG_path))
        self.netD.load_state_dict(torch.load(self.netD_path))
        print("Checkpoints loaded successfuly")
         

def main(args):
    tac_gan = TACGAN(args)
    tac_gan.train()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--continue-training', action='store_true')
    parser.add_argument('--netg-path', type=str, default='')
    parser.add_argument('--netd-path', type=str, default='')
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--data-root', type=str, default='')
    parser.add_argument('--dataset', type=str, default='flowers')
    parser.add_argument('--num-cls', type=int, default=102)
    parser.add_argument('--save-dir', type=str, default='checkpoints/')
    parser.add_argument('--save-prefix', type=str, default='')
    parser.add_argument('--save-after', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    main(args)
