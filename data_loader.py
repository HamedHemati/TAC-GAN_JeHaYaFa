import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy.random as random


class ImTextDataset(Dataset):
    '''
    data_dir   : path to the directory that contains the dataset files
    dataset    : name of the dataset (eg. flowers/coco)
    train      : determines which part of the dataset to use. By default:train
    image_size : intented image size. By default: 128x128
    '''
    def __init__(self, data_dir, dataset='flowers', train=True, image_size=128):
        super(ImTextDataset, self).__init__()
        
        self.train = train  # determines whether to return train or validation images
        self.data_dir = data_dir
        self.dataset = dataset
        self.image_dir = os.path.join(self.data_dir, dataset, 'jpg')
        self.train_ids = []  # list of training ids, i.e. file names of the images
        self.val_ids = []  # list of validation ids, i.e. file names of the images
        self.captions = {}  # list with 5 captions per id
        self.captions_encoded = {}  # array with the encoded captions per id. array shape is (5, 4800)
        self.classes = {}  # one-hot vector encoding the class
        self.trans_img = transforms.Compose([transforms.Scale(image_size), transforms.CenterCrop(image_size),
                                             transforms.ToTensor(),]) # transformation for output image 
        self._load_pickle_files()

    def __getitem__(self, index):
        if self.train:
            id = self.train_ids[index]
        else:
            id = self.val_ids[index]
        # load the image and apply the transformation to it
        image_path = os.path.join(self.image_dir, id)
        image =Image.open(image_path)
        image = self.trans_img(image)
        # pick a random encoded caption
        rnd = random.randint(0,5)
        rnd_encoded_caption = self.captions_encoded[id][rnd]
        return image, self.classes[id], rnd_encoded_caption, self.captions[id][rnd]

    def __len__(self):
        if self.train:
            return len(self.train_ids)
        else:
            return len(self.val_ids)

    def _load_pickle_files(self):
        # generate file paths
        path = os.path.join(self.data_dir, self.dataset)
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
