import os
from collections import defaultdict

import random

from torch.utils.data import Dataset

from .enums import TRAIN_CITIES, VAL_CITIES, TEST_CITIES

class CityScapesDataset(Dataset):

    def __init__(self, annotations_dir:str, original_images_dir:str, dataset_type:str="train"):

        self.annotations_dir = annotations_dir
        self.original_images_dir = original_images_dir
        self.dataset_type = dataset_type

        if self.dataset_type == "train":
            self.image_file_paths = self.get_image_file_paths(TRAIN_CITIES)
        
        elif self.dataset_type == "val":
            self.image_file_paths = self.get_image_file_paths(VAL_CITIES)

        elif self.dataset_type == "test":
            self.image_file_paths = self.get_image_file_paths(TEST_CITIES)

    def __len__(self):        
        return len(self.image_file_paths)

    def __getitem__(self, idx):        
        batch_images = self.image_file_paths[idx]
            
        city, image_file = batch_images
        image_id = '_'.join(image_file.split('_')[:-1])
        gtfine_image_file = f'{image_id}_gtFine_labelIds.png'

        return {
            "original_image_path":f'{self.original_images_dir}/{city}/{image_file}',
            "gtfine_image_path":f'{self.annotations_dir}/{city}/{gtfine_image_file}'
        }

    def get_image_file_paths(self, cities):

        image_file_paths = [] 

        for city in cities:
            image_files = os.listdir(f'{self.original_images_dir}/{city}')
            image_file_paths.extend([(city, image_file) for image_file in image_files])
            # image_ids = ['_'.join(image_file.split('_')[:-1]) for image_file in image_files]
            # gtfine_files = [f'{image_id}_gtFine_labelIds.png' for image_id in image_ids]

        random.shuffle(image_file_paths)

        return image_file_paths