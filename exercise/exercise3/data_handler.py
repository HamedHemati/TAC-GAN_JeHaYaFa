from os import listdir
from os.path import join
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


# returns list of images and correspoinding classes + list of labels
def getImageList(root):
    # list of folders in the dataset root
    dirs = listdir(root)
    # label names
    labels = [d.split('-',1)[1] for d in dirs]
    # list of image paths and corresponding classes
    images = []
    classes = []
    # add images and corresponding classes in each folder to the image list
    for i,d in enumerate(dirs):
        path = join(root,d)
        ims = [join(path, im) for im in listdir(path)]
        clss = [i]*len(ims)
        images += ims
        classes += clss
    return images, classes, labels


# returns train/test/val sets given the image list
def getSetLists(images, classes):
    train = [] # list of training images
    train_c = [] # list of corresponding classes of training images
    test = []
    test_c = []
    val = []
    val_c = []
    temp =[]
    temp_c = []
    # split the data for train/(test/val) with ratio 7:3
    sff = StratifiedShuffleSplit(n_splits=2, test_size=0.3)
    for tr, tmp in sff.split(images, classes):
        # set the elements of train set
        train = [images[i] for i in tr]
        train_c = [classes[i] for i in tr]
        # set the rest for test and val
        temp = [images[i] for i in tmp]
        temp_c = [classes[i] for i in tmp]
        # split the remaining for test/val with ratio 2:1
        sff = StratifiedShuffleSplit(n_splits=2, test_size=0.36)
        for tst, vl in sff.split(temp, temp_c):
            test = [temp[i] for i in tst]
            test_c = [temp_c[i] for i in tst]
            val = [temp[i] for i in vl]
            val_c = [temp_c[i] for i in vl]
            break
        break
    return train, train_c, test, test_c, val, val_c


class DogDS(Dataset):
    def __init__(self, im_list, im_classes, imgSize):
        self.im_list = im_list
        self.im_classes = im_classes
        self.transform = transforms.Compose([transforms.Scale(imgSize),
                                             transforms.CenterCrop(imgSize),
                                             transforms.ToTensor()])

    def __getitem__(self, index):
        image = Image.open(self.im_list[index])
        image = self.transform(image)
        
        return image, self.im_classes[index]

    def __len__(self):
        return len(self.im_list)
     
        
def getDatasets(dataset_root, imgSize):
    images, classes, labels = getImageList(dataset_root)
    train, train_c, test, test_c, val, val_c = getSetLists(images, classes)
    trainSet = DogDS(train, train_c, imgSize)
    testSet = DogDS(test, test_c, imgSize)
    valSet = DogDS(val, val_c, imgSize)
    print('loaded the datasets successfuly')
    return trainSet, testSet, valSet
