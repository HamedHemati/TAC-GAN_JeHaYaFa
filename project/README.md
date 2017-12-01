# TAC-GAN

This is a TAC-GAN (https://arxiv.org/abs/1703.06412) implementation in PyTorch.

## Requirements
TBD: add requirements

 - Python 3.5
 - PyTorch ???
 - Theano 0.9.0 : for skip thought vectors
 - scikit-learn : for skip thought vectors
 - NLTK 3.2.1 : for skip thought vectors
 
## Flowers Dataset
To train the TAC-GAN on the flowers dataset, download the dataset by
doing the following.

1. Download the flower images from
[here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz).
Extract the ```102flowers.tgz``` file and copy the extracted ```jpg``` folder
 to ```Data/datasets/flowers```

2. Download the captions from
[here](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/).
Extract the downloaded file, copy the text_c10 folder and paste it in ```
Data/datasets/flowers``` directory

3. Download the pretrained skip-thought vectors model from
[here](https://github.com/ryankiros/skip-thoughts#getting-started) and copy
the downloaded files to ```Data/skipthoughts```

4. Run the flowers dataset preparation
    
    ```
    python dataprep_flowers.py --data_dir=Data --dataset=flowers
    ```

    This script will create a set of pickled files in the `Data` directory which
    will be used during training. The following are the available flags for data preparation:
    
    FLAG | VALUE TYPE | DEFAULT VALUE | DESCRIPTION
    --- | --- | --- | ---
    data_dir | str | Data | The data directory |
    dataset | str | flowers | Dataset to use. For Eg., "flowers" |

## COCO Dataset
To train the TAC-GAN on the COCO dataset, download the dataset by
doing the following.

 1. Download the dataset from http://cocodataset.org/
     - 2017 Train images: http://images.cocodataset.org/zips/train2017.zip
     - 2017 Train/Val annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
 2. Extract both archives to a directory of your choice, e.g. `Data`
 3. Run the COCO dataset preparation
    
    ```
    python dataprep_coco.py --data_dir=Data --dataset=coco
    ```