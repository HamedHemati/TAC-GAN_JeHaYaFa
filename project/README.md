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

## COCO Dataset

 1. Download the dataset from http://cocodataset.org/
     - 2017 Train images: http://images.cocodataset.org/zips/train2017.zip
     - 2017 Train/Val annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
 2. Extract both archives to a directory of your choice, e.g. `Data`
 3. Run the COCO dataset preparation
    
    ```
    python dataprep_coco.py --data_dir=Data --dataset=coco
    ```