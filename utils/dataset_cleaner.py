from os import listdir,remove
from os.path import join
from scipy.misc import imread


def removeGrayImages(directory):
    fileList = listdir(directory)
    count = 0
    l = len(fileList)
    lst = []
    for i,file in enumerate(fileList):
        path = join(directory,file)
        img = imread(path)
        print('checking '+str(i)+'/'+str(l))
        if len(img.shape) != 3:
            remove(path)
            count+=1
            lst.append(path)
            print('image removed')
            
    print('number of removals: '+str(count))
    print('these files were removed:')
    print(lst)


# just change the input directory
removeGrayImages('coco/jpg')


