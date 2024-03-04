from typing import Any
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

from torchvision.utils import save_image

from PIL import Image

import cv2
import numpy as np

from .enums import Label, LABELS_LIST
from typing import List, DefaultDict

class BicubicInterpolation(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        img = F.to_pil_image(img)  # Convert the input tensor to PIL Image
        img = F.resize(img, self.output_size, interpolation=Image.BICUBIC)
        img = F.to_tensor(img)  # Convert back to a PyTorch tensor
        return img

class CityScapesCollateFn(object):

    def __init__(self, 
                image_resize=(512, 512),
                interpolation_strategy="bilinear_interpolation",
                split="train", 
                crop=True,
                is_transform=True):

        self.resizing_width, self.resizing_height = image_resize
        self.split = split
        self.interpolation_strategy = interpolation_strategy

        # self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        
        # these are 19
        # self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
        # ]
        # self.classes = [7, 8, 11, 12, 19, 22, 23, 24, 26]

        # based on CategoryId
        self.classes = list(range(7))

        # these are 19 + 1; "unlabelled" is extra        
        self.crop = crop 

        # for void_classes; useful for loss function
        
        # dictionary of valid classes 7:0, 8:1, 11:2
        # self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

        # self.ignore_index = len(self.class_map) + 1
        self.ignore_index = 250
        # dictionary of valid classes 0:7, 1:8, 2:11
        # self.id_to_class_map = dict(zip(range(19), self.valid_classes))

        self._initialize_labels()

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

        self.transforms = transforms.Compose([
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor(),
            transforms.Resize((self.resizing_width, self.resizing_height), transforms.InterpolationMode.BILINEAR)
            
        ])

    def transform(self, image_arr:np.array, label_arr:np.array):

        image_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)

        # print(image_arr.shape)

        if self.crop:
            image_arr = image_arr[:800,:,:]
            label_arr = label_arr[:800, :]

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

        if self.split == "train":
            return {
                "image_tensors" : batch_images_tensors, #(bs, 3, h=512, w=1024)
                "label_tensors" : batch_label_tensors #(bs)
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