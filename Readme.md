# Part Affinity Field Implementation in PyTorch

Pure python and pytorch implementation of [
Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050). Original caffe implementation is [here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) 

## COCO Multi-person Dataset and Dataloader Setup:
Download  train2017.zip, val2017.zip and annotations_trainval2017.zip from [COCO Project](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm) The keypoints description can be found [here](http://cocodataset.org/#format-data). Extract the folders and place them in '/data'. Pre-processing of the dataset is done on the fly. To visualize the data loader use:
 
 ```python visualize_coco_data.py -data ../data -vizPaf```

The data loader depends on pycocoapi which can be installed using 

```pip install pycocotools```

Design choices at this stage are i) Width of part affinity field ii) Heatmap std. iii) Choosing the parts for PAF iv) PAF magnitude (smooth/rigid) v) Masking crowded/unannotated joints? Due to differences in scale of the persons across dataset, some of these choices play an important role during training. Original paper uses constant PAF with a single part width for all joints across dataset. But this can introduce a lot of noise to the data in terms of misleading ground truth pafs/heatmaps. Alternate design choices are exposed in this implementation while keeping the original choices as default. 

## NN Model:
The paper uses first 10 layers from VGG-19 as feature extractor followed by 7 heatmap/paf regression stages with intermediate supervision at each stage. The same is implemented here.

## Training and Testing:

```python main.py -data ../data -expID vgg19 -model vgg19 -train```

Comprehensive list of opts can be found in ```opts/``` folder. To debug/visualize each image's outputs during training ```-vizOut``` flag is helpful. 50k iterations takes around 11.5 hours with a batch size of 8 on a GTX 1080 GPU 

![Sample Output](output/sample_heatmap.png?raw=true "Sample heatmap outpus")