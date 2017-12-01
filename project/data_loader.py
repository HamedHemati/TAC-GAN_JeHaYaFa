import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms  as transforms


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


class PickleDataset(Dataset):

    def __init__(self, data_dir: str, dataset: str):
        super(PickleDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset = dataset



class flowers(Dataset):
    def __init__(self, labeled_img_list, labels, captions):
        self.__data = labeled_img_list
        self.__labels = labels
        self.__captions = captions
        self.__transform = flowers.__transform()

    def __getitem__(self, index):
        img = self.__transform(Image.open(self.__data[index][0]))
        lab = self.__labels.index(self.__data[index][1])
        cap = self.__captions.index(self.__data[index[2]])
        return img, lab, cap

    def __len__(self):
        return len(self.__data)

    def resolve_label(self, index):
        return self.__labels[index]

    def get_label(self, index):
        return self.__data[index][1]

    def get_labels(self):
        return self.__labels

    def resolve_caption(self, index):
        return self.__captions[index]

    def get_caption(self, index):
        return self.__data[index][2]

    def get_captions(self):
        return self.__captions

    def get_unique_captions(self):
        return set(self.__captions)

    def get_data(self):
        return self.__data

    def get_full_img_name(self, index):
        return self.__data[index][0]

    @staticmethod
    def __transform():
        return transforms.Compose([
            transforms.Scale((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,))
        ])
