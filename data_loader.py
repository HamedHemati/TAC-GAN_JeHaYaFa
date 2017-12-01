import os
import pickle

import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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
    """
    Usage:
    > dataset = PickleDataset('/home/fbalsiger/Documents/data', 'flowers')
    """

    def __init__(self, data_dir: str, dataset: str, train: bool=True):
        super(PickleDataset, self).__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.image_dir = os.path.join(self.data_dir, 'datasets', dataset, 'jpg')

        self.train = train  # determines whether to return train or validation images

        self.train_ids = []  # list of training ids, i.e. file names of the images
        self.val_ids = []  # list of validation ids, i.e. file names of the images
        self.captions = {}  # list with 5 captions per id
        self.captions_encoded = {}  # array with the encoded captions per id. array shape is (5, 4800)
        self.classes = {}  # one-hot vector encoding the class

        self._load_pickle_files()

    def __getitem__(self, index):
        if self.train:
            id = self.train_ids[index]
        else:
            id = self.val_ids[index]

        image_path = os.path.join(self.image_dir, id)
        image = skimage.io.imread(image_path)

        return image, self.classes[id], self.captions_encoded[id], self.captions[id]

    def __len__(self):
        if self.train:
            return len(self.train_ids)
        else:
            return len(self.val_ids)

    def _load_pickle_files(self):
        # generate file paths
        path = os.path.join(self.data_dir, 'datasets', self.dataset)

        train_ids_file = os.path.join(path, 'train_ids.pkl')
        val_ids_file = os.path.join(path, 'val_ids.pkl')
        captions_file = os.path.join(path, self.dataset + '_caps.pkl')
        captions_encoded_file = os.path.join(path, self.dataset + '_tv.pkl')
        classes_file = os.path.join(path, self.dataset + '_tc.pkl')

        # load pickle files
        self.train_ids = pickle.load(open(train_ids_file, 'rb'))
        self.val_ids = pickle.load(open(val_ids_file, 'rb'))
        self.captions = pickle.load(open(captions_file, 'rb'))
        self.captions_encoded = pickle.load(open(captions_encoded_file, 'rb'))
        self.classes = pickle.load(open(classes_file, 'rb'))

    # Yannick had this transform implemented and applied it to the loaded image...?
    # @staticmethod
    # def _transform():
    #     return transforms.Compose([
    #         transforms.Scale((128, 128)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (1.0,))
    #     ])
