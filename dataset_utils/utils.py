
import torch
import torch.nn.functional as F 
import os

import numpy as np

from PIL import Image
from dataset_utils.enums import CLASS_ID_TO_COLOR_CODE, VISUALIZATION_FILES

def plot_segmentation_map(predicted_segmentation_map:torch.tensor, label_segmentation_map:torch.tensor,gtfine_image_paths:list, class_map:dict, output_dir:str):

    for idx, gtfine_image_path in enumerate(gtfine_image_paths):

        city_name, image_name = gtfine_image_path.split('/')[-2], gtfine_image_path.split('/')[-1]
        root_image = image_name.split('_gtFine')[0]

        segmenetation_map = predicted_segmentation_map[idx] 
        true_map = label_segmentation_map[idx]

        image_h, image_w = segmenetation_map.shape 

        segmenetation_map = segmenetation_map.view(-1)
        
        segmenetation_map = [torch.tensor(class_map[predicted_id.item()]) for predicted_id in segmenetation_map]
        segmenetation_map = torch.stack(segmenetation_map)

        true_map = true_map.view(-1)
        true_map = [torch.tensor(class_map[predicted_id.item()]) for predicted_id in true_map]
        true_map = torch.stack(true_map)

        segmenetation_map = segmenetation_map.reshape(image_h, image_w, -1)
        true_map = true_map.reshape(image_h, image_w, -1)
        
        if not os.path.exists(f'{output_dir}/{city_name}'):
            os.makedirs(f'{output_dir}/{city_name}')

        segmenetation_map = segmenetation_map.to(torch.int32)
        segmentation_map_arr = segmenetation_map.numpy()

        segmentation_image = Image.fromarray(segmentation_map_arr.astype(np.uint8))
        segmentation_image.save(f'{output_dir}/{city_name}/{root_image}_prediction_segmentation_color.png')

        true_map = true_map.to(torch.int32)
        true_map_arr = true_map.numpy()

        true_image = Image.fromarray(true_map_arr.astype(np.uint8))
        true_image.save(f'{output_dir}/{city_name}/{root_image}_true_segmentation_color.png')

    torch.cuda.empty_cache()
        

def calculate_soft_iou_loss(prediction_tensor:torch.tensor, target_tensor:torch.tensor, output_classes:int, smooth=1e-5):
    target_one_hot_tensor = F.one_hot(target_tensor, output_classes).permute(0, 3, 1, 2)
    
    intersection = torch.sum(prediction_tensor * target_one_hot_tensor, dim=(2, 3))
    union = torch.sum(prediction_tensor.pow(2) + target_one_hot_tensor, dim=(2, 3)) + smooth

    soft_iou = (intersection + smooth)/union
    loss = 1 - soft_iou

    return loss.mean()

def calculate_focal_loss(cross_entropy_loss:torch.tensor ,gamma=2, alpha=None, reduction='mean'):
    pt = torch.exp(-cross_entropy_loss)

    focal_loss = (1 - pt)**gamma * cross_entropy_loss

    if reduction == "mean":
        return focal_loss.mean()

    elif reduction == "sum":
        return focal_loss.sum()
    
    else:
        return focal_loss

    
def calculate_iou(prediction_tensor:torch.tensor, target_tensor:torch.tensor, num_classes:int):

    prediction_tensor = torch.argmax(prediction_tensor, dim=1)  # Convert probabilities to class indices
    target_tensor = target_tensor.long()    

    iou_per_class = torch.zeros(num_classes)

    for class_label in range(num_classes):
        intersection = torch.sum((prediction_tensor == class_label) & (target_tensor == class_label))
        union = torch.sum((prediction_tensor == class_label) | (target_tensor == class_label))
        iou_per_class[class_label] = intersection.float() / union.float()

    miou = torch.mean(iou_per_class)

    return iou_per_class, miou        

def calculate_iou_for_classes(prediction_tensor: torch.Tensor, target_tensor: torch.Tensor, class_indices: list):

    prediction_tensor = torch.argmax(prediction_tensor, dim=1)  # Assuming dim 0 is the class probability
    target_tensor = target_tensor.long()
    
    ious = []
    
    for class_idx in class_indices:
        intersection = torch.sum((prediction_tensor == class_idx) & (target_tensor == class_idx))
        union = torch.sum((prediction_tensor == class_idx) | (target_tensor == class_idx))
        # Avoid division by zero
        iou = intersection.float() / union.float() if union.float() != 0 else torch.tensor(0.0)
        ious.append(iou)
    
    iou_per_class = torch.tensor(ious)
    mean_iou = torch.mean(iou_per_class) if len(ious) > 0 else torch.tensor(0.0)
    
    return iou_per_class, mean_iou

def calculate_dice_loss(prediction_tensor:torch.tensor, target_tensor:torch.tensor, num_classes:int, smooth=1.0):

    prediction_tensor = F.sigmoid(prediction_tensor)
    
    target_tensor = F.one_hot(target_tensor, num_classes=num_classes).permute(0, 3, 1, 2)

    intersection = (prediction_tensor * target_tensor).sum()
    union = prediction_tensor.sum() + target_tensor.sum()

    dice_coeff = (2.0 * intersection + smooth)/(union + smooth)
    dice_loss = 1.0 - dice_coeff.mean()    

    return dice_loss

def calculate_pixel_level_accuracy(prediction_tensor:torch.tensor, target_tensor:torch.tensor, num_classes:int):

    prediction_tensor = torch.argmax(prediction_tensor, dim=1)  # Convert probabilities to class indices
    target_tensor = target_tensor.long()    

    # Calculate pixel-level accuracy
    correct_pixels = torch.sum(prediction_tensor == target_tensor)
    total_pixels = torch.numel(target_tensor)
    total_pixel_accuracy = correct_pixels.float() / total_pixels

    class_accuracies = []
    for class_idx in range(num_classes):
        correct_pixels = torch.sum((prediction_tensor == class_idx) & (target_tensor == class_idx))
        incorrect_pixels = torch.sum((prediction_tensor != class_idx) & (target_tensor == class_idx))

        acc_class = correct_pixels/(correct_pixels + incorrect_pixels)
        class_accuracies.append(acc_class)

    return torch.tensor(class_accuracies), total_pixel_accuracy

def calculate_accuracy_for_classes(prediction_tensor: torch.Tensor, target_tensor: torch.Tensor, class_indices: list):

    prediction_tensor = torch.argmax(prediction_tensor, dim=1)
    target_tensor = target_tensor.long()
    
    accuracies = []
    
    for class_idx in class_indices:
        correct_pixels = torch.sum((prediction_tensor == class_idx) & (target_tensor == class_idx))
        total_class_pixels = torch.sum(target_tensor == class_idx)
        # Avoid division by zero
        accuracy = correct_pixels.float() / total_class_pixels.float() if total_class_pixels.float() != 0 else torch.tensor(0.0)
        accuracies.append(accuracy)
    
    accuracy_per_class = torch.tensor(accuracies)
    mean_accuracy = torch.mean(accuracy_per_class) if len(accuracies) > 0 else torch.tensor(0.0)
    
    return accuracy_per_class, mean_accuracy


def convert_time_to_readable_format(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    if not hour:
        if not minutes:
            time = f"{seconds} Seconds"
        else:
            time = f"{minutes} minute/s and {seconds} seconds"
    else:
        time = f"{hour} hour/s {minutes} minute/s {seconds} seconds"

    return time
