import os
import argparse
import traceback
import pickle
import random
import time

import numpy as np
from os.path import join

import skipthoughts
from pycocotools.coco import COCO


def get_one_hot_targets(target: list):
    n_target = len(target)
    lbl_idxs = np.arange(n_target)
    one_hot_targets = np.zeros((n_target, n_target))
    one_hot_targets[np.arange(n_target), lbl_idxs] = 1

    return target, one_hot_targets, n_target


def get_one_hot_targets_from_file(target_file_path):
    try:
        with open(target_file_path) as f:
            target = f.readlines()
            target = [t.strip('\n') for t in target]
            return get_one_hot_targets(target)
    except IOError:
        print('Could not load the labels.txt file in the dataset. A '
              'dataset folder is expected in the "data/datasets" '
              'directory with the name that has been passed as an '
              'argument to this method. This directory should contain a '
              'file called labels.txt which contains a list of labels and '
              'corresponding folders for the labels with the same name as '
              'the labels.')
        traceback.print_stack()


def one_hot_encode_str_lbl(lbl, target, one_hot_targets):
    """
    Encodes a string label into one-hot encoding

    Example:
        input: "window"
        output: [0 0 0 0 0 0 1 0 0 0 0 0]
    the length would depend on the number of classes in the dataset. The
    above is just a random example.

    :param lbl: The string label
    :return: one-hot encoding
    """
    idx = target.index(lbl)
    return one_hot_targets[idx]


def save_caption_vectors_coco(data_dir):
    """Prepares the COCO dataset.

    :param data_dir: The path to the data directory. Needs to contain a jpg directory with the images and
        an annotations directory with the JSON files.
    :type data_dir: str
    """

    data_type = 'train2017'
    instances_file = '{}/annotations/instances_{}.json'.format(join(data_dir, 'coco'), data_type)
    captions_file = '{}/annotations/captions_{}.json'.format(join(data_dir, 'coco'), data_type)

    coco = COCO(instances_file)
    coco_caps = COCO(captions_file)

    cats = coco.loadCats(coco.getCatIds())
    class_names = [cat['name'] for cat in cats]
    # TODO: filter class names?

    # make one hot
    target, one_hot_targets, n_target = get_one_hot_targets(class_names)

    image_captions = {}
    image_classes = {}

    # get all images belonging to class and store in image_classes and image_captions
    for class_name in class_names:
        cat_id = coco.getCatIds(class_name)
        img_ids = coco.getImgIds(catIds=cat_id)

        image_class = one_hot_encode_str_lbl(class_name, target, one_hot_targets)

        for img_id in img_ids:
            img_file_name = '%.12d.jpg' % img_id
            image_classes[img_file_name] = image_class

            ann_ids = coco_caps.getAnnIds(imgIds=img_id)
            annotations = coco_caps.loadAnns(ann_ids)
            image_captions[img_file_name] = [annotation['caption'] for annotation in annotations][0:5]

    encode_and_save(image_captions, image_classes, data_dir, 'coco')


def save_caption_vectors_flowers(data_dir, dt_range=(1, 103)):
    target_file_path = os.path.join(data_dir, "flowers/allclasses.txt")
    caption_dir = join(data_dir, 'flowers/text_c10')
    image_captions = {}
    image_classes = {}
    class_dirs = []
    class_names = []

    target, one_hot_targets, n_target = get_one_hot_targets_from_file(target_file_path)

    for i in range(dt_range[0], dt_range[1]):  # for each class
        class_dir_name = 'class_%.5d' % i
        class_dir = join(caption_dir, class_dir_name)
        class_names.append(class_dir_name)
        class_dirs.append(class_dir)
        onlyimgfiles = [f[0:11] + ".jpg" for f in os.listdir(class_dir) if 'txt' in f]
        for img_file in onlyimgfiles:
            image_classes[img_file] = None
            image_captions[img_file] = []

    for class_dir, class_name in zip(class_dirs, class_names):
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for i, cap_file in enumerate(caption_files):
            if i % 50 == 0:
                print(str(i) + ' captions extracted from' + str(class_dir))
            with open(join(class_dir, cap_file)) as f:
                str_captions = f.read()
                captions = str_captions.split('\n')
            img_file = cap_file[0:11] + ".jpg"

            # 5 captions per image
            image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]
            image_classes[img_file] = one_hot_encode_str_lbl(class_name,
                                                             target,
                                                             one_hot_targets)

    encode_and_save(image_captions, image_classes, data_dir, 'flowers')


def encode_and_save(image_captions, image_classes, data_dir: str, dataset: str):
    model = skipthoughts.load_model()
    encoded_captions = {}
    for i, img in enumerate(image_captions):
        st = time.time()
        encoded_captions[img] = skipthoughts.encode(model, image_captions[img])
        if i % 20 == 0:
            print(i, len(image_captions), img)
            print("Seconds", time.time() - st)

    img_ids = list(image_captions.keys())

    random.shuffle(img_ids)
    n_train_instances = int(len(img_ids) * 0.9)
    tr_image_ids = img_ids[0:n_train_instances]
    val_image_ids = img_ids[n_train_instances:-1]

    pickle.dump(image_captions, open(os.path.join(data_dir, dataset, dataset + '_caps.pkl'), "wb"))

    pickle.dump(tr_image_ids, open(os.path.join(data_dir, dataset, 'train_ids.pkl'), "wb"))
    pickle.dump(val_image_ids, open(os.path.join(data_dir, dataset, 'val_ids.pkl'), "wb"))

    ec_pkl_path = join(data_dir, dataset, dataset + '_tv.pkl')
    pickle.dump(encoded_captions, open(ec_pkl_path, "wb"))

    fc_pkl_path = join(data_dir, dataset, dataset + '_tc.pkl')
    pickle.dump(image_classes, open(fc_pkl_path, "wb"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data',
                        help='Data directory')
    parser.add_argument('--dataset', type=str, default='flowers',
                        help='Dataset to use. "flowers" or "coco"')
    args = parser.parse_args()

    dataset_dir = join(args.data_dir, "datasets")
    if args.dataset == 'flowers':
        save_caption_vectors_flowers(dataset_dir)
    if args.dataset == 'coco':
        save_caption_vectors_coco(dataset_dir)
    else:
        print('Preprocessor for this dataset is not available.')


if __name__ == '__main__':
    main()
