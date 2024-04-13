from typing import Any
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from torchvision.utils import save_image

from PIL import Image

import cv2
import numpy as np

from typing import List, DefaultDict
from .enums import Label, LABELS_LIST, DatasetParams

class ActiveLearningCollateFn(object):

    def __init__(self, 
                image_resize=(512, 1024),
                interpolation_strategy="bilinear_interpolation",
                split="train", 
                crop=True, 
                is_transform=True):
        
        self.resizing_width, self.resizing_height = image_resize
        self.split = split
        self.interpolation_strategy = interpolation_strategy 

        self.classes = list(range(7))     
        self.ignore_index = 250

        self._initialize_labels()

        self._create_non_overlapping_region_masks()

    def _initialize_labels(self):
        
        self.labels = []

        for label_info in LABELS_LIST:
            label = Label(tuple(label_info))
            self.labels.append(label)

        self.category_id_2_label = DefaultDict(list)
        self.category_id_2_color = {}

        for label in self.labels:
            self.category_id_2_label[label.category_id].append(label)

        for category_id, label_list in self.category_id_2_label.items():
            
            for label in label_list:
                self.category_id_2_color[category_id] = label.color

        self.valid_classes = list(self.category_id_2_color.keys())

    def _create_non_overlapping_region_masks(self):
        
        num_rows = self.resizing_height // DatasetParams.crop_size[0]
        num_cols = self.resizing_width // DatasetParams.crop_size[1] 

        self.crops = []

        for row in range(num_rows):
            for col in range(num_cols):
                start_x = col * DatasetParams.crop_size[1]
                start_y = row * DatasetParams.crop_size[0]
                end_x = start_x + DatasetParams.crop_size[1]
                end_y = start_y + DatasetParams.crop_size[0]
                self.crops.append([(start_x, start_y), (end_x, end_y)])  

    def transform(self, image_arr:np.array, label_arr:np.array):

        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)

        # print(image_arr.shape)

        # if self.crop:
        #     image_arr = image_arr[:800,:,:]
        #     label_arr = label_arr[:800, :]

        for category_id, label_list in self.category_id_2_label.items():
            for label in label_list:                
                id = label.id
                label_arr[label_arr == id] = category_id 

        if self.interpolation_strategy == "bilinear_interpolation":
            image_arr = cv2.resize(image_arr, (self.resizing_height, self.resizing_width), interpolation=cv2.INTER_LINEAR)
            label_arr = cv2.resize(label_arr, (self.resizing_height, self.resizing_width), interpolation=cv2.INTER_LINEAR)
        
        elif self.interpolation_strategy == "lanczos_interpolation":
            image_arr = cv2.resize(image_arr, (self.resizing_height, self.resizing_width), interpolation=cv2.INTER_LANCZOS4)
            label_arr = cv2.resize(label_arr, (self.resizing_height, self.resizing_width), interpolation=cv2.INTER_LANCZOS4)

        elif self.interpolation_strategy == "bicubic_interpolation":
            image_arr = cv2.resize(image_arr, (self.resizing_height, self.resizing_width), interpolation=cv2.INTER_CUBIC)
            label_arr = cv2.resize(label_arr, (self.resizing_height, self.resizing_width), interpolation=cv2.INTER_CUBIC)
                
        image_arr = np.transpose(image_arr, (2, 0, 1))

        return image_arr, label_arr

    def preprocess_images(self, batch_images):

        batch_images_tensors, batch_label_tensors = [], []
        original_image_paths, gtfine_image_paths = [], []

        for item_dict in batch_images:
            original_image_path = item_dict["original_image_path"]
            gtfine_image_path = item_dict["gtfine_image_path"]

            original_image = cv2.imread(original_image_path)
            gtfine_image = cv2.imread(gtfine_image_path, cv2.IMREAD_UNCHANGED)

            image_arr, label_arr = self.transform(
                original_image, gtfine_image
            )

            image_arr = torch.from_numpy(image_arr).float()
            label_arr = torch.from_numpy(label_arr).long()

            batch_images_tensors.append(image_arr)
            batch_label_tensors.append(label_arr)

            original_image_paths.append(original_image_path)
            gtfine_image_paths.append(gtfine_image_path)

        batch_images_tensors = torch.stack(batch_images_tensors, dim=0)
        batch_label_tensors = torch.stack(batch_label_tensors, dim=0)

        if self.split == "state":
            return {
                "image_tensors" : batch_images_tensors, #(bs, 3, h=512, w=1024)
                "label_tensors" : batch_label_tensors #(bs, h=512, w=1024)
            }

        elif self.split == "reward":
            return {
                "image_tensors" : batch_images_tensors, #(bs, 3, h=512, w=1024)
                "label_tensors" : batch_label_tensors #(bs, h=512, w=1024)
            }
        
        elif self.split == "train":
            return {
                "image_tensors" : batch_images_tensors, #(bs, 3, h=512, w=1024)
                "label_tensors" : batch_label_tensors #(bs, h=512, w=1024)
            }

        else:
            return {
                "image_tensors" : batch_images_tensors, 
                "label_tensors" : batch_label_tensors,
                "original_image_paths": original_image_paths,
                "gtfine_image_paths": gtfine_image_paths
            }            

    def __call__(self, batch_images):
        return self.preprocess_images(batch_images)              