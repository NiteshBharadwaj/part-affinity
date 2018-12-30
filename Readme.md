# Part Affinity Field Implementation in PyTorch

Pure python and pytorch implementation of [
Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050). Original caffe implementation is [here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) 

## COCO Multi-person Dataset and Dataloader Setup:
Download  train2017.zip, val2017.zip and annotations_trainval2017.zip from [COCO Project](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm) The keypoints description can be found [here](http://cocodataset.org/#format-data). Extract the folders and place them in '/data'. Pre-processing the dataset is done on the fly. To visualize the data loader use:
 
 ```python visualize_coco_data.py -vizPaf```

The data loader depends on pycocoapi which can be installed using 

```pip install pycocotools```

Design choices at this stage are i) Width of part affinity field ii) Heatmap width iii) Choosing the parts for PAF etc. iv) PAF magnitude (smooth/rigid) v) Handling occluded joints. Due to differences in scale of the persons across dataset, these choices play an important role during training. Original paper uses constant PAF with a single part width for all joints across dataset. But this can introduce a lot of noise to the data. Alternate design choices are exposed in this implementation while keeping the original choices as default.

## NN Model:
Original paper uses VGG-19 with first few layers fixed and an intermediate training. Stacked hourglass and Resnet area also  explored in this implementation.

## Training:

```python main.py  -expID ```
Code to be uploaded soon.