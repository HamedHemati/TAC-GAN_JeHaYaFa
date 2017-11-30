import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from os import listdir
from os import path
import numpy as np

def get_one_hot_targets(target_file_path):
	target = []
	one_hot_targets = []
	n_target = 0
	try :
		with open(target_file_path) as f :
			target = f.readlines()
			target = [t.strip('\n') for t in target]
			n_target = len(target)
	except IOError :
		print('Could not load the labels.txt file in the dataset. A '
		      'dataset folder is expected in the "data/datasets" '
		      'directory with the name that has been passed as an '
		      'argument to this method. This directory should contain a '
		      'file called labels.txt which contains a list of labels and '
		      'corresponding folders for the labels with the same name as '
		      'the labels.')
		traceback.print_stack()

	lbl_idxs = np.arange(n_target)
	one_hot_targets = np.zeros((n_target, n_target))
	one_hot_targets[np.arange(n_target), lbl_idxs] = 1

	return target, one_hot_targets, n_target

def one_hot_encode_str_lbl(lbl, target, one_hot_targets):
        '''
        Encodes a string label into one-hot encoding
        Example:
            input: "window"
            output: [0 0 0 0 0 0 1 0 0 0 0 0]
        the length would depend on the number of labels in the dataset. The
        above is just a random example.
        :param lbl: The string label
        :return: one-hot encoding
        '''
        idx = target.index(lbl)
        return one_hot_targets[idx]

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

    @staticmethod
    def get_sets(root='Data/datasets/flowers/', get_full_set=False):
        class_range =(1, 103)
        img_dir = path.join(root, 'jpg')
        all_caps_dir = path.join(root, 'all_captions.txt')
        target_file_path = path.join(root, "allclasses.txt")
        caption_dir = path.join(root, 'text_c10')

        image_files = [f for f in listdir(img_dir) if 'jpg' in f]

        if get_full_set:
            return_set = []
        else:
            return_set = {'train': [], 'valid': [], 'test': []}
        labels = []

        image_captions = {}
        image_labels = {}
        label_dirs = []
        label_names = []
        #img_ids = []
        imgs = listdir(img_dir)

        target, one_hot_targets, n_target = get_one_hot_targets(target_file_path)

        for i in range(class_range[0], class_range[1]):
            label_dir_name = 'class_%.5d' % (i)
            label_dir = path.join(caption_dir, label_dir_name)
            label_names.append(label_dir_name)
            label_dirs.append(label_dir)
            onlyimgfiles = [f[0:11] + ".jpg" for f in listdir(label_dir)
                            if 'txt' in f]
            for img_file in onlyimgfiles:
                image_labels[img_file] = None

            for img_file in onlyimgfiles:
                image_captions[img_file] = []

        for label_dir, label_name in zip(label_dirs, label_names):
            caption_files = [f for f in listdir(label_dir) if 'txt' in f]
            for i, cap_file in enumerate(caption_files):
                if i % 50 == 0:
                    print(str(i) + ' captions extracted from' + str(label_dir))
                with open(path.join(label_dir, cap_file)) as f:
                    str_captions = f.read()
                    captions = str_captions.split('\n')
                img_file = cap_file[0:11] + ".jpg"

                # 5 captions per image
                image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]
                image_labels[img_file] = one_hot_encode_str_lbl(label_name,
                                                                 target,
                                                                 one_hot_targets)
        img_ids = list(image_captions.keys())

        if not get_full_set:
            train_upper_b = int(0.7*len(imgs))
            valid_upper_b = int(train_upper_b + 0.2*len(imgs))
        for idi, i in enumerate(imgs):
            append_val = [path.join(img_dir, i), image_labels[i], image_captions[i]]

            if get_full_set:
                return_set.append(append_val)
            else:
                if idi < train_upper_b:
                    add_to = 'train'
                elif idi < valid_upper_b:
                    add_to = 'valid'
                else:
                    add_to = 'test'
                return_set[add_to].append([path.join(img_dir, i), image_labels[i], image_captions[i]])
        if get_full_set:
            return flowers(return_set, labels, captions)
        else:
            return (
                flowers(return_set['train'], labels, captions),
                flowers(return_set['valid'], labels, captions),
                flowers(return_set['test'], labels, captions)
            )


if __name__ == '__main__':
    big_set = flowers.get_sets(get_full_set=True)

    print(big_set.get_label(2))
    print(big_set.get_caption(2))
    print(big_set.get_full_img_name(2))
