from os import listdir,remove
from os.path import join
from scipy.misc import imread
import os.path
import pickle

def removeGrayImages(directory, dataset='coco'):

    image_dir = join(directory, 'jpg')
    fileList = listdir(image_dir)

    count = 0
    lst = []
    for i,file in enumerate(fileList):
        path = join(image_dir, file)
        img = imread(path)
        # print('checking '+str(i)+'/'+str(l))
        if len(img.shape) != 3:
            # remove(path)
            count+=1
            lst.append(file)
            print(file)
            
    print('number of gray images: ' + str(count))
    print('these files were removed:')
    print(lst)

    # load pickles
    train_ids_file = os.path.join(directory, 'train_ids.pkl')
    val_ids_file = os.path.join(directory, 'val_ids.pkl')
    captions_file = os.path.join(directory, dataset + '_caps.pkl')
    captions_encoded_file = os.path.join(directory, dataset + '_tv.pkl')
    classes_file = os.path.join(directory, dataset + '_tc.pkl')

    # load pickle files
    train_ids = pickle.load(open(train_ids_file, 'rb'))
    val_ids = pickle.load(open(val_ids_file, 'rb'))
    captions = pickle.load(open(captions_file, 'rb'))
    captions_encoded = pickle.load(open(captions_encoded_file, 'rb'))
    classes = pickle.load(open(classes_file, 'rb'))

    for key in lst:
        train_ids.remove(key)
        val_ids.remove(key)
        del captions[key]
        del captions_encoded[key]
        del classes[key]

    pickle.dump(train_ids, open(train_ids_file, "wb"))
    pickle.dump(val_ids, open(val_ids_file, "wb"))
    pickle.dump(captions, open(captions_file, "wb"))
    pickle.dump(captions_encoded, open(captions_encoded_file, "wb"))
    pickle.dump(classes, open(classes_file, "wb"))


# just change the input directory
removeGrayImages('/home/fbalsiger/Documents/data/datasets/coco')


