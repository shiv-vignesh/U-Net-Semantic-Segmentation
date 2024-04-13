import os, json
from collections import defaultdict
import random

from torch.utils.data import Dataset

from .enums import DatasetParams

import sys
sys.path.append('../')
from dataset_utils.enums import TRAIN_CITIES, VAL_CITIES, TEST_CITIES

class ActiveLearningDataset(Dataset):
    def __init__(self, annotations_dir:str, original_images_dir:str, dataset_type:str="train", partition_dir:str="al_partition"):
        
        self.annotations_dir = annotations_dir
        self.original_images_dir = original_images_dir
        self.dataset_type = dataset_type     

        self.get_data_partition(partition_dir) 

        if self.dataset_type == "train":
            self.image_file_paths = self.training_dataset
        
        elif self.dataset_type == "state":
            self.image_file_paths = self.state_dataset

        elif self.dataset_type == "reward":
            self.image_file_paths = self.reward_dataset

        elif self.dataset_type == "validation":
            self.image_file_paths = self.validation_dataset                    

    def get_data_partition(self, partition_dir:str):
        ''' 
        TODO
        Create a method to split 
        - State must match with the same distribution as overall train split.  
        '''
        if not os.path.exists(f'{partition_dir}/split_files.json'):
            print(f'Stratifying Images')
            self.random_stratify_images(f'{partition_dir}/split_files.json')

        else:
            print(f'Loading Stratified Images')
            partition_json = json.load(open(f'{partition_dir}/split_files.json'))
            self.state_dataset = partition_json["state_dataset"]
            self.reward_dataset = partition_json["reward_dataset"]
            self.training_dataset = partition_json["training_dataset"]  
            self.validation_dataset = partition_json["validation_dataset"]
        
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

    def random_stratify_images(self, filename:str):        
        remaining_images = self.image_file_paths
        
        self.state_dataset = random.sample(remaining_images, DatasetParams.num_DS_images)
        remaining_images = [image for image in remaining_images if image not in self.state_dataset]

        self.reward_dataset = random.sample(remaining_images, DatasetParams.num_DR_images)
        remaining_images = [image for image in remaining_images if image not in self.reward_dataset]

        self.training_dataset = random.sample(remaining_images, DatasetParams.num_DT_images)
        remaining_images = [image for image in remaining_images if image not in self.training_dataset]

        self.validation_dataset = random.sample(remaining_images, DatasetParams.num_DV_images)
        remaining_images = [image for image in remaining_images if image not in self.validation_dataset]        

        with open(filename, 'w+') as f:
            json.dump({
                "state_dataset":self.state_dataset,
                "reward_dataset":self.reward_dataset,
                "training_dataset":self.training_dataset,
                "validation_dataset":self.validation_dataset
            }, f)

    def get_image_file_paths(self, cities):

        image_file_paths = [] 

        for city in cities:
            image_files = os.listdir(f'{self.original_images_dir}/{city}')
            image_file_paths.extend([(city, image_file) for image_file in image_files])

        random.shuffle(image_file_paths)

        return image_file_paths
    
    