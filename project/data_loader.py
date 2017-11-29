import torch
from torch.utils.data import Dataset
import numpy as np
import  torchvision.transforms  as transforms


class DumbDataset(Dataset):
    def __init__(self, num_images=200, num_classes=100):
        super(DumbDataset, self).__init__()
        self.num_images = num_images
        self.num_classes = num_classes
        #self.trans = transforms.Compose([])
        self.images = torch.rand(self.num_images, 3, 64, 64)
        self.labels = torch.from_numpy(np.random.randint(0,self.num_classes, self.num_images))
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)