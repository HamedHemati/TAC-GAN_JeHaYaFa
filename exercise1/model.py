''''
Very Deep Learning Course
Assignment 1
Group Name: JeHaYaFa
'''
import torch.nn as nn
import torch.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        #MNIST input size 28x28
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(2,2), padding=(0,0)) #padding valid
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(2,2), stride=(1,1), padding=(1,1)) #padding same
        self.maxPool = nn.MaxPool2d((2,2), (1,1)) 
        self.dropout2d = nn.Dropout2d(p=0.35)
        self.fc1 = nn.Linear(5408, 256)
        self.dropout1d = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
    
    def forward(self, x): 
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))   
        x = self.dropout2d(self.maxPool(x)) 
        #size after droupout 13*13*32=5408
        x = x.view(x.size(0), 5408)
        x = self.dropout1d(self.tanh(self.fc1(x)))
        x = self.softmax(self.fc2(x))
        
        return x
        
