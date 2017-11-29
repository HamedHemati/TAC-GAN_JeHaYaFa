import argparse
from time import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import  Dataset, DataLoader
from model import NetD, NetG
import numpy as np
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms

class TACGAN():
    def __init__(self, args):
        # class variables
        self.lr = args.lr
        self.cuda = args.use_cuda
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.epochs = args.epochs
        self.nz = 100 # size of the noise vector
        self.num_classes = 10
        self.netD = NetD(num_classes=self.num_classes)
        self.netG = NetG(nz=self.nz+self.num_classes)
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.NLLLoss()
        self.trainset_loader = None
        self.evalset_loader = None  
        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix
        self.netG_path = args.netg_path
        self.netD_path = args.netd_path
        self.data_root = args.data_root

        # convert to cuda tensors
        if self.cuda and torch.cuda.is_available():
            print('cuda is enabled')
            self.netD = self.netD.cuda()
            self.netG = self.netG.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.ce_loss = self.ce_loss.cuda() 

        # optimizers for both netD and netG
        self.optimizerD = optim.Adam(params=self.netD.parameters(), lr=self.lr)
        self.optimizerG = optim.Adam(params=self.netG.parameters(), lr=self.lr)

        # create dir for saving checkpoints if does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    # start training process
    def train(self):
        # load trainset and evalset
        train_set = dset.CIFAR10(root=self.data_root, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(self.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        self.trainset_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True, num_workers=2)
        
        print("dataset loaded successfuly")
        # repeat for the number of epochs
        for epoch in range(self.epochs):
            self.trainEpoch(epoch)
            #self.evalEpoch(epoch)
            self.saveCheckpoint(epoch)

    # train epoch
    def trainEpoch(self, epoch):
        self.netD.train() # set to train mode
        self.netG.train() # set to train mode  <<<<<<<<<<<<<< check this
        
        netd_loss_sum = 0
        netg_loss_sum = 0
        start_time = time()
        for i, (images, labels) in enumerate(self.trainset_loader):
            self.netD.zero_grad()       
            ########## Update NetD ##########
            # train with real data
            batch_size = images.size(0) # !batch size my be different (from self.batch_size) for the last batch
            images, labels = Variable(images), Variable(labels) # !labels should be LongTensor
            lbl_real = Variable(torch.ones(batch_size, 1))
            lbl_fake = Variable(torch.zeros(batch_size, 1))
            if self.cuda:
                images, labels = images.cuda(), labels.cuda()
                lbl_real, lbl_fake = lbl_real.cuda(), lbl_fake.cuda()
            
            outD_real, outC_real = self.netD(images)
            lossD_real = self.bce_loss(outD_real, lbl_real)
            lossC_real = self.ce_loss(outC_real, labels)

            # train with fake data
            noise = Variable(torch.rand(batch_size, self.nz)) # create random noise
            onehot_labels = np.zeros((batch_size, self.num_classes))
            onehot_labels[np.arange(batch_size), list(labels.data)] = 1 #! use random labels other than the labels of the real data
            onehot_labels = Variable(torch.from_numpy(onehot_labels).type(torch.FloatTensor)) # create the correspoing one_hot vector for the random labels
            if self.cuda:
                noise, onehot_labels = noise.cuda(), onehot_labels.cuda()
            
            # concat noise and one_hot vector for the correspoing class
            noise = torch.cat([noise, onehot_labels], 1)
            noise = noise.view(batch_size, self.nz+self.num_classes, 1,1)

            fake = self.netG(noise)
            outD_fake, outC_fake = self.netD(fake.detach())
            lossD_fake = self.bce_loss(outD_fake, lbl_fake)
            lossC_fake = self.ce_loss(outC_fake, torch.max(onehot_labels, 1)[1].view(batch_size))
            # backward and forwad for NetD
            netD_loss = lossC_real + lossC_fake + lossD_real + lossD_fake
            netD_loss.backward()
            self.optimizerD.step()      
    
            ########## Update NetG ##########
            # train with fake data
            self.netG.zero_grad()
            d_fake, c_fake = self.netD(fake)
            lossD_fake_G = self.bce_loss(d_fake, lbl_real)
            lossC_fake_G = self.ce_loss(c_fake, torch.max(onehot_labels, 1)[1].view(batch_size))
            netG_loss = lossD_fake_G + lossC_fake_G 
            netG_loss.backward()
            self.optimizerG.step()
            
            netd_loss_sum += netD_loss.data[0]
            netg_loss_sum += netG_loss.data[0]
            ### print progress info ###
            print('epoch %d/%d, %.2f%% completed. Loss_D: %.4f, Loss_G: %.4f'
                  %(epoch, self.epochs,(float(i)/len(self.trainset_loader))*100, netD_loss.data[0], netG_loss.data[0]))
       
        end_time = time()
        epoch_time = (end_time-start_time)/60
        netd_avg_loss = netd_loss_sum / len(self.trainset_loader)
        netg_avg_loss = netg_loss_sum / len(self.trainset_loader)
        log_msg = '-------------------------------------------\n'
        log_msg += 'epoch %d took %.2f minutes\n'%(epoch, epoch_time)
        log_msg += 'NetD average loss: %.4f, NetG average loss: %.4f\n' %(netd_avg_loss, netg_avg_loss)
        log_msg += '-------------------------------------------\n\n'
        print(log_msg)
        with open(os.path.join(self.save_dir, 'training_log'),'a') as log_file:
            log_file.write(log_msg)

    # eval epoch                   
    def evalEpoch(self, epoch):
        self.netD.eval()
        self.netG.eval()
        #for i (images,labels) in enumerate(self.trainset_loader):

    # save after each epoch
    def saveCheckpoint(self, epoch):
        name_netD = "netD_" + self.save_prefix + "_epoch_" + str(epoch) + ".pth"
        name_netG = "netG_" + self.save_prefix + "_epoch_" + str(epoch) + ".pth"
        torch.save(self.netD.state_dict(), os.path.join(self.save_dir, name_netD))
        torch.save(self.netG.state_dict(), os.path.join(self.save_dir, name_netG))
        print("checkpoints for epoch %d saved successfuly" %(epoch))

    # load checkpoints to continue training
    def loadCheckpoint(self):
        self.netG.load_state_dict(torch.load(self.netG_path))
        self.netD.load_state_dict(torch.load(self.netD_path))
        print("checkpoints loaded successfuly")
         

def main(args):
    tac_gan = TACGAN(args)
    tac_gan.train()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--resume-training', action='store_true')
    parser.add_argument('--netg-path', type=str, default='')
    parser.add_argument('--netd-path', type=str, default='')
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--data-root', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--save-prefix', type=str, default='')
    args = parser.parse_args()
    main(args)
