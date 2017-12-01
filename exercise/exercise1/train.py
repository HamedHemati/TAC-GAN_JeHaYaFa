''''
Very Deep Learning Course
Assignment 1
Group Name: JeHaYaFa
'''
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import argparse
from model import Model
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def train(epoch, args, model, criterion, train_loader, optimizer):
    model.train()
    epoch_loss = 0
    for idx, (data,target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        print('{} percent of train epoch {} is done, err = {}'.format(int(float(idx)/len(train_loader)*100), epoch, loss.data[0]))   
    print('train epoch {} finished, avg. loss = {}'.format(epoch, float(epoch_loss)/len(train_loader)))
    print('------------------\n')
    return float(epoch_loss)/len(train_loader)

def test(epoch, args, model, criterion, test_loader):   
    model.eval()
    epoch_loss = 0

    for idx, (data,target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        epoch_loss += loss.data[0]
        print('{} percent of test epoch {} is done, err = {}'.format(int(float(idx)/len(test_loader)*100), epoch, loss.data[0]))   
        
    print('test epoch {} finished, avg. loss = {}'.format(epoch, float(epoch_loss)/len(test_loader)))
    print('------------------\n')
    return float(epoch_loss)/len(test_loader)

def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, 
                                                transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))])),
                                                batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader =  torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True, 
                                                transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))])),
                                                batch_size=args.batch_size, shuffle=True, num_workers=2)                                               
    model = Model()
    if args.cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.optimizer=='ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer=='ADADELTA':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    #start epochs
    train_err = []
    test_err = []
    for i in range(args.epochs):
        train_err.append(train(i, args, model, criterion, train_loader, optimizer))
        test_err.append(test(i, args, model, criterion, test_loader))
        #draw and save the plot for train and test error
        plt.plot(train_err, color='red', label='training error')
        plt.plot(test_err, color='blue', label='test error', linestyle='--')
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.legend(loc='best')
        plt.savefig(args.file_name)
        plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default="SGD")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--file-name', type=str, default="plot.png")
    args = parser.parse_args()
    main(args)