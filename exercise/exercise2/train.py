from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import data_handler
from model import getDogNet
import matplotlib.pyplot as plt
import pickle

def weightInit(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(epoch, args, train_set, tr_size, model, criterion, optimizer):
    model.train()
    epoch_loss = 0
    matches = 0

    for idx, (data,target) in enumerate(train_set):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()   
        output = model(data)
        loss = criterion(output, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        

        #get number of matches
        matches += torch.sum(torch.max(output, 1)[1]==target).data[0]
        print('{}{} of train epoch {} is done, err = {}'.format(int(float(idx)/len(train_set)*100), '%', epoch, loss.data[0]))   
        print('number of matches: ' + str(matches) + '\n')
    print('train epoch {} finished, avg. loss = {}, accuracy = {}'.format(epoch, float(epoch_loss)/len(train_set), float(matches)/tr_size))
    print('------------------\n')
    return float(epoch_loss)/len(train_set), float(matches)/tr_size


def validate(epoch, args, val_set, val_size, model, criterion):   
    model.eval()
    epoch_loss = 0
    matches = 0

    for idx, (data,target) in enumerate(val_set):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = criterion(output, target)
        epoch_loss += loss.data[0]

        #get number of matches
        matches += torch.sum(torch.max(output, 1)[1]==target).data[0]
        print('{}{} of val epoch {} is done, err = {}'.format(int(float(idx)/len(val_set)*100),'%', epoch, loss.data[0]))   
        print('number of matches: ' + str(matches) + '\n')

    print('val epoch {} finished, avg. loss = {}, accuracy = {}'.format(epoch, float(epoch_loss)/len(val_set),  float(matches)/val_size))
    print('------------------\n')
    return float(epoch_loss)/len(val_set), float(matches)/val_size


def main(args):
    tr, te, val = data_handler.getDatasets(imgSize=224)
    train_set = DataLoader(dataset=tr, batch_size=args.batchSize, num_workers=3, shuffle=True)
    test_set = DataLoader(dataset=te, batch_size=args.batchSize, num_workers=3, shuffle=True)
    val_set = DataLoader(dataset=val, batch_size=args.batchSize, num_workers=3, shuffle=True)
    
    tr_size = len(tr)
    te_size = len(te)
    val_size = len(val)

    model = getDogNet(pre_trained=args.preTrained, checkpoint_path=args.checkPoint)
    criterion = nn.NLLLoss()
    
    if args.cuda:
        model = model.cuda()
        #criterion = criterion.cuda()

    ##########################3
    model.apply(weightInit)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    train_err = []
    train_acc = []
    val_err = []
    val_acc =[]
    
    for i in range(args.epochs):
        train_e, train_a = train(i, args, train_set, tr_size, model, criterion, optimizer)
        val_e, val_a = validate(i, args, val_set, val_size, model, criterion)
        train_err.append(train_e)
        train_acc.append(train_a)
        val_err.append(val_e)
        val_acc.append(val_a)

        #draw and save the plot for train and test error
        plt.plot(train_err, color='red', label='training error')
        plt.plot(val_err, color='blue', label='val error')
        plt.plot(train_acc, color='pink', label='train accuracy', linestyle='--')
        plt.plot(val_acc, color='green', label='val accuracy', linestyle='--')
        plt.xlabel('epoch')
        plt.ylabel('error/accuracy')
        plt.legend(loc='best')
        plt.savefig(args.fileName)
        plt.close()
        
    err_acc_file = "err_acc.txt"
    checkpoint_file = "checkpoints/dognet.pth"
    if args.preTrained:
        err_acc_file = "err_acc_pretrained.txt"
        checkpoint_file = "checkpoints/dognet_pretrained.pth"
    # save checkpoint
    torch.save(model, checkpoint_file)

    # save err/acc info to a file    
    with open(err_acc_file, "wb") as fp:
        pickle.dump(train_err, fp) 
        pickle.dump(train_acc, fp)
        pickle.dump(val_err, fp)
        pickle.dump(val_acc, fp)       

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fileName', type=str, default="plot.png")
    parser.add_argument('--checkPoint', type=str, default='checkpoints/alexnet.pth')
    parser.add_argument('--preTrained', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
