import torch
from model import DogNet
import data_handler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle
import numpy as np


def loadTestData(dataset_path):
    _,_,tst = data_handler.getDatasets(dataset_root=dataset_path, imgSize=224)
    testSet = DataLoader(dataset=tst, batch_size=1, num_workers=4, shuffle=True)
    print('Test set loaded successfuly with ' + str(len(testSet)) + ' elements')
    return testSet
    

def loadModel(checkpoint_path):
    dogNet = torch.load(checkpoint_path, map_location=lambda storage, loc:storage)
    print('Model loaded successfuly')
    return dogNet


def getFeatures(dog_net, test_set):
    dog_net.eval()
    correct = []
    correct_labels = []
    false = []
    false_labels = []
    l = len(test_set)
    for idx, (data,target) in enumerate(test_set):
        data,target = Variable(data), Variable(target)
        output = dog_net(data)
        max_val = torch.max(output, 1)[1].data.numpy()[0][0]
        tar_val = target.data.numpy()[0]
        if max_val==tar_val:
            correct.append(dog_net.getFeat(data).data.numpy()[0])
            correct_labels.append(int(target.data.numpy()))
        elif max_val!=tar_val :
            #switch back to feats
            false.append(dog_net.getFeat(data).data.numpy()[0])
            false_labels.append(int(target.data.numpy()))
        print(str(idx) + '/' + str(l) + ' evaluated')
        if idx==500:
            break    

    print('# of correct matches: '+str(len(correct)) + ', correct labels: '+str(len(correct_labels)))   
    print('# of false matches: '+str(len(false)) + ', false labels: '+str(len(false_labels)))   
    return correct, correct_labels, false, false_labels


def main():
    checkpoint_path = 'checkpoints/dognet.pth'
    dataset_path = '/home/hamed/Repositories/VDL/Assignment 2/DogsDS'
    dog_net = loadModel(checkpoint_path)
    test_set = loadTestData(dataset_path)
    correct, correct_labels, false, false_labels = getFeatures(dog_net, test_set)
    with open('features','wb') as fp:
        pickle.dump(correct, fp)
        pickle.dump(correct_labels,fp)
        pickle.dump(false,fp)
        pickle.dump(false_labels,fp)


if __name__ == '__main__':
    main()

