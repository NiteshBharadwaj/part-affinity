import os

import cv2
import torch.utils.data as data
from pycocotools.coco import COCO

from .coco_process_utils import clean_annot, get_ignore_mask, get_heatmap, get_paf, get_keypoints


class CocoDataSet(data.Dataset):
    def __init__(self, data_path, split='train'):
        self.coco_year = 2017
        self.coco = COCO(os.path.join(data_path, 'annotations/person_keypoints_{}{}.json'.format(split, self.coco_year)))
        self.split = split
        self.data_path = data_path
        self.do_augment = split == 'train'

        # load annotations that meet specific standards
        self.indices = clean_annot(self.coco, data_path, split)
        self.img_dir = os.path.join(data_path, split + str(self.coco_year))

    def get_item_raw(self, index):
        index = self.indices[index]
        anno_ids = self.coco.getAnnIds(index)
        annots = self.coco.loadAnns(anno_ids)
        img_path = os.path.join(self.img_dir, self.coco.loadImgs([index])[0]['file_name'])
        img = self.load_image(img_path)
        keypoints = get_keypoints(self.coco, img, annots)
        heat_map = get_heatmap(self.coco, img, keypoints)
        paf = get_paf(self.coco, img, keypoints)
        ignore_mask = get_ignore_mask(self.coco, img, annots)
        return img, heat_map, paf, ignore_mask, keypoints

    def __getitem__(self, index):
        img, heat_map, paf, ignore_mask, _ = self.get_item_raw(index)
        return img, heat_map, paf, ignore_mask


    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = img.astype('float32') / 255.
        return img

    def __len__(self):
        return len(self.indices)




