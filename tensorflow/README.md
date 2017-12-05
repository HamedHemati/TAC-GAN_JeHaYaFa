# TAC-GAN in TensorFlow
The scripts in this directory are modified scripts from the original [TensorFlow implementation](https://github.com/dashayushman/TAC-GAN). 
The modification affects only the dataset part, i.e. we added a COCO dataset loader. See our main README for introductions on how to prepare the dataset.

1. Replace the files in the TensorFlow implementation by the files in this directory
2. Run the training
    1. To reproduce the paper results on the flowers dataset:
    
            python train.py --t_dim=100 --image_size=128 --data_set=flowers --model_name=TAC_GAN --data_dir=Data --train=True --resume_model=True --z_dim=100 --n_classes=102 --epochs=400 --save_every=20 --caption_vector_length=4800 --batch_size=128
            
    2. To run the TAC-GAN on the COCO dataset:
            
            python train.py --t_dim=100 --image_size=128 --data_set=coco --model_name=TAC_GAN --data_dir=Data --train=True --resume_model=True --z_dim=100 --n_classes=80 --epochs=400 --save_every=20 --caption_vector_length=4800 --batch_size=128