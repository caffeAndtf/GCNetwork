# Geometry and Context Network

### This project implements GC Network developed by Kendal, et al(2017)

### Data used for training model: 
    SceneFlow_Driving_DrivingFinalPass

### Preprocessing:
    We crop training patches with size of 256x256 (different from that in the paper) from training images and normalize each channel.

### Two ways to download Driving dataset: 
    1. create subdirectories sceneflow/driving in data, download and tar driving_final pass and driving_disparity from <here>. 
    
    2. You can also run the command “sh download.sh”, which will create subdirectories and download datasets.

#### Train the model by running:
    python train.py
 
#### (Optional) Specify the pretrained weight by
    1. set it in train_params.pywhile 
    2. python train.py -wpath <path to the pretrained weight>

#### To enable training with Monkaa dataset, uncomment the relevant snippet in train.py.

### Reference :
    Kendall, Alex, et al. "End-to-End Learning of Geometry and Context for Deep Stereo Regression." arXiv preprint arXiv:1703.04309 (2017).