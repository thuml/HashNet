# HashNet
Code release for "HashNet: Deep Learning to Hash by Continuation" (ICCV 2017) 

## Datasets
We use ImageNet, NUS-WIDE and COCO dataset in our experiments. You can download the ImageNet dataset [here]() and NUS-WIDE dataset [here]().
As for COCO dataset, we use COCO 2014, you can download the dataset [here](http://mscoco.org/dataset/#download).
After downloading, you need to move the imagenet.tar.gz to ./data/imagenet and extract the file there. Also, you need to move the nus_wide.tar.gz to ./data/nuswide_81 and extract the file there. For COCO dataset, you can modify the list file in ./data/coco/ to your local dataset directory.

Each line follows the following format:
```
image_name label_represented_by_0_and_1
```
## Training
```
dataset_name = imagenet, nuswide_81 or coco
./build/tools/caffe train -solver models/train/dataset_name/solver.prototxt -weights ./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel -gpu gpu_id
```

## Evaludation
```
dataset_name = imagenet, nuswide_81 or coco
python models/predict/dataset_name/predict_parallel.py --gpu gpu_id --model_path your_caffemodel_path --save_path the_path_to_save_your_code
```
