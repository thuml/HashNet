# HashNet
PyTorch implementation for ["HashNet: Deep Learning to Hash by Continuation" (ICCV 2017)](https://arxiv.org/abs/1702.00758) 

## Prerequisites
Linux or OSX

NVIDIA GPU + CUDA (may CuDNN) and corresponding PyTorch framework (version 0.3.1)

Python 2.7/3.5

## Datasets
We use ImageNet, NUS-WIDE and COCO dataset in our experiments. You can download the ImageNet dataset and NUS-WIDE dataset [here](https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?usp=sharing).
As for COCO dataset, we use COCO 2014, which can be downloaded [here](http://mscoco.org/dataset/#download). And in case of COCO changes in the future, we also provide a download link [here](https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?usp=sharing) on google drive.
After downloading, you need to move the imagenet.tar.gz to [./data/imagenet](./data/imagenet) and extract the file there.
```
mv imagenet.tar.gz ./data/imagenet
cd ./data/imagenet
tar -zxvf imagenet.tar.gz
```
Also, for NUS-WIDE, you need to move the nus_wide.tar.gz to [./data/nuswide_81](./data/nuswide_81) and extract the file there. 
```
mv nus_wide.tar.gz ./data/nus_wide
cd ./data/nus_wide
tar -zxvf nus_wide.tar.gz
```
For COCO dataset, you need to extract both train and val archive for COCO in [./data/coco](./data/coco).
If you download from [COCO download page](http://mscoco.org/dataset/#download),
```
mv train2014.zip ./data/coco
mv val2014.zip ./data/coco
cd ./data/coco
unzip train2014.zip
unzip val2014.zip
```
If you use our shared [link](https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?usp=sharing)
```
mv coco.tar.gz ./data/coco
cd ./data/coco
tar -zxvf coco.tar.gz
unzip train2014.zip
unzip val2014.zip
```

You can also modify the list file(txt format) in ./data as you like. Each line in the list file follows the following format:
```
<image path><space><one hot label representation>
```

## Training
First, you can manually download the PyTorch pre-trained model introduced in `torchvision' library or if you have connected to the Internet, you can automatically downloaded them.
Then, you can train the model for each dataset using the followling command.
```
cd src
python train.py --gpu_id 0 --dataset coco --prefix resnet50_hashnet --hash_bit 48 --net ResNet50 --lr 0.0003 --class_num 1.0
```
You can set the command parameters to switch between different experiments. 
- "gpu_id" is the GPU ID to run experiments.
- "hash_bit" parameter is the number of bits of the hash codes.
- "dataset" is the dataset selection. In our experiments, it can be "imagenet", "nus_wide" or "coco".
- "prefix" is the path to output model snapshot and log file in "snapshot" directory.
- "net" sets the base network. For details of setting, you can see network.py.
    - For AlexNet, "net" is AlexNet.    
    - For VGG Net, "net" is like VGG16. Detail names are in network.py.
    - For ResNet, "net" is like ResNet50. Detail names are in network.py.
- "lr" is the learning rate.
- "class_num" is the positive and negative pairs balance weight.

## Evaluation
You can evaluate the Mean Average Precision(MAP) result on each dataset using the followling command.
```
cd src
python test.py --gpu_id 0 --dataset coco --prefix resnet50_hashnet --hash_bit 48 --snapshot iter_09000
```
You can set the command parameters to switch between different experiments. 
- "gpu_id" is the GPU ID to run experiments.
- "hash_bit" parameter is the number of bits of the hash codes.
- "dataset" is the dataset selection. In our experiments, it can be "imagenet", "nus_wide" or "coco".
- "prefix" is the path to output model snapshot and log file in "snapshot" directory.
- "snapshot" is the snapshot model name. "iter_09000" means the model snapshoted at iteration 9000.
