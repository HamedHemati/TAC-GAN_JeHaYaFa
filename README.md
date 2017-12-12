# TAC-GAN_JeHaYaFa

This repository contains exercises and the project for the Very Deep Learning lecture at the University of Fribourg, Switzerland.
Our group consists of:

 - Ajayi Jesutofunmi
 - Balsiger Fabian
 - Hemati Hamed
 - Murena Patrick
 - Suter Yannick

# Project: TAC-GAN

A [TAC-GAN](https://arxiv.org/abs/1703.06412) implementation in PyTorch. The original TensorFlow implementation can be found [here](https://github.com/dashayushman/TAC-GAN).
Following files were copied from the original implementation:

 - `dataprep_flowers.py`: renamed to `data_prep.py` and modified to include the COCO dataset
 - `encode_text.py`
 - `skipthoughts.py`
 - `train.py`: see the `tensorflow` directory

## Requirements
The project requires Python 3.5.2. 
Install all other requirements by running `pip install -r requirements.txt`.

## Data Preparation
We assume that we store the datasets in a `Data` directory in the root directory of this project throughout this data preparation.

Independent of the dataset, the skip-thought vectors need to be downloaded.
 
 1. Download the [pre-trained skip-thought vectors model](https://github.com/ryankiros/skip-thoughts#getting-started) 
into the directory `Data/skipthoughts` by executing:
    
        wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
        wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
        wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
        wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
        wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
        wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
        wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
    
 2. Adjust the `path_to_models` and `path_to_labels` (lines 23 and 24) in `skipthoughts.py` if you use another data directory than `Data`.
 3. Install the required tokenizers with a Python script:
    
        import nltk
        nltk.download('punkt')

### Flowers Dataset
To train the TAC-GAN on the flowers dataset, download the dataset by
doing the following.

1. Download the flower images from
[here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz).
Extract the `102flowers.tgz` file and copy the extracted `jpg` folder
 to `Data/datasets/flowers`.

2. Download the captions from
[here](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/).
Extract the `text_c10` folder and the `allclasses.txt` and paste it in the `Data/datasets/flowers` directory.

4. Run the flowers dataset preparation:
    
    ```
    python data_prep.py --data_dir=Data --dataset=flowers
    ```

    This script will create a set of pickled files in the `Data/datasets/flowers` directory which
    will be used during training.

ATTENTION: The TensorFlow implementation will raise an error when the flowers dataset is prepared with this implementation.
It is necessary to change the names of two pickle files in the `Data/datasets/flowers` directory:

 - `flowers_tc.pkl` to `flower_tc.pkl`
 - `flowers_tv.pkl` to `flower_tv.pkl`

### COCO Dataset
To train the TAC-GAN on the COCO dataset, download the dataset by
doing the following.

 1. Download the [COCO dataset](http://cocodataset.org/)
     - [2017 Train images](http://images.cocodataset.org/zips/train2017.zip)
     - [2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
 2. Extract both archives to the folder `Data/datasets/coco`.
 3. Rename the folder `train2017` to `jpg`.
 4. Run `make` in the directory `pycocotools`.
 5. Run the COCO dataset preparation:
    
    ```
    python data_prep.py --data_dir=Data --dataset=coco
    ```
    
    This script will create a set of pickled files in the `Data/datasets/coco` directory which
    will be used during training.

## Experiments

### COCO Dataset
We perform three experiments with the COCO dataset:

 1. Full dataset
 2. 9 animal categories: ['cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
 3. 10 random categories: ['wine glass', 'cup', 'keyboard', 'cat', 'banana', 'surfboard', 'bus', 'truck', 'baseball glove', 'microwave']

