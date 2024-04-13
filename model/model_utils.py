import random, math, torch

import numpy as np

import torch.nn.functional as F
from dataset_utils.enums import RLParams, DatasetParams
from .dqn import DQN

def create_mask_tensors(selected_regions):
    mask = torch.zeros(DatasetParams.resized_image_resolution[0], DatasetParams.resized_image_resolution[1])

    for region in selected_regions:
        start_x, start_y = region[0]
        end_x, end_y = region[1]

        mask[start_x:end_x, start_y:end_y] = 1

    return mask

def calculate_crop_region_accuracy(prediction_crop:torch.tensor, label_crop:torch.tensor, classes:list):

    accuracy_per_class = {}

    for cls in classes:
        correct = ((prediction_crop.argmax(dim=1) == cls) & (label_crop == cls)).sum().item()
        total = (label_crop == cls).sum().item()
        accuracy = correct / total if total != 0 else 0  # Avoid division by zero        

        accuracy_per_class[cls] = accuracy

    return torch.tensor(list(accuracy_per_class.values()))

def calculate_crop_region_iou(prediction_crop: torch.tensor, label_crop: torch.tensor, classes: list):
    iou_per_class = {}

    for cls in classes:
        intersection = ((prediction_crop.argmax(dim=1) == cls) & (label_crop == cls)).sum().item()
        union = ((prediction_crop.argmax(dim=1) == cls) | (label_crop == cls)).sum().item()
        iou = intersection / union if union != 0 else 0  # Avoid division by zero

        iou_per_class[cls] = iou

    return torch.tensor(list(iou_per_class.values()))

def calculate_crop_region_entropy(prediction_crop:torch.tensor):

    prob_dist = F.softmax(prediction_crop, dim=1)
    entropy = -torch.sum(prob_dist * torch.log2(prob_dist + 1e-20), dim=1)  # Add a small epsilon to prevent log(0)

    average_entropy = entropy.mean(dim=0)

    return average_entropy    

def calculate_label_distribution(label_crop:torch.tensor, classes:list):

    label_crop_distribution = []

    for crop in label_crop:
        flattened_crop = crop.flatten()
        counts = torch.bincount(flattened_crop, minlength=len(classes))

        crop_distribution = counts.float()/counts.sum()      
        label_crop_distribution.append(crop_distribution)  

    label_crop_distribution = torch.stack(label_crop_distribution, dim=0)
    label_crop_distribution = torch.mean(label_crop_distribution, dim=0)

    return label_crop_distribution

def compute_kl_divergence(training_crop:torch.tensor, state_crop:torch.tensor, epsilon = 1e-12):
    ''' 
    modification - 
    1. weighting the KL divergence by the inverse frequency of classes in the dataset, thereby amplifying the signal from underrepresented classes.
    2. KL Divergence only for underrepresented classes. Calculate the KL_Div on each region for just underrep classes
    '''
    
    #epsilon to avoid div by 0. 

    training_crop_normalized = F.softmax(training_crop, dim=1)
    state_crop_normalized = F.softmax(state_crop, dim=1)    
    kl_div = training_crop_normalized * torch.log((training_crop_normalized + epsilon) / (state_crop_normalized + epsilon))

    return kl_div.sum(dim=1) 

def compute_reward_weights(reward_per_class_acc:torch.tensor, reward_per_class_iou:torch.tensor, minority_class_indices):

    inverse_accuracy = 1.0 / reward_per_class_acc[minority_class_indices]
    inverse_iou = 1.0 / reward_per_class_iou[minority_class_indices]

    normalized_weights_acc = inverse_accuracy / inverse_accuracy.sum() * len(minority_class_indices)
    normalized_weights_iou = inverse_iou / inverse_iou.sum() * len(minority_class_indices)

    weights = (normalized_weights_acc + normalized_weights_iou) / 2

    return weights

def select_action(state_label_distribution:torch.tensor, training_label_distribution:torch.tensor, 
                  state_accuracy_tensor:torch.tensor, state_iou_tensor:torch.tensor, 
                  state_entropy_tensor:torch.tensor, steps_done:int, policy_net:DQN):

    ''' 
    state_label_distribution - (128, num_classes)
    training_label_distribution - (len(training_dataset), 128, num_classes)

    state_accuracy_tensor - (128, num_classes)
    state IOU - (128, num_classes)
    entropy - (128, 64, 64); needs to be reshaped to (128, 4096) before DQN. 

    kl_divergence - (150, 128)

    selected_training_regions - (150, 8). Indices between (0, 127) of the selected 128 regions.

    '''

    training_kl_div = []

    for label_distribution in training_label_distribution:
        kl_div = compute_kl_divergence(label_distribution, state_label_distribution)
        training_kl_div.append(kl_div)

    training_kl_div = torch.stack(training_kl_div, dim=0) #(150, 128)

    sample = random.random()
    eps_threshold = RLParams.EPS_END + (RLParams.EPS_START - RLParams.EPS_END) * \
        math.exp(-1. * steps_done / RLParams.EPS_DECAY)

    steps_done += 1    

    num_regions = state_entropy_tensor.size(0)

    state_accuracy_tensor = state_accuracy_tensor.unsqueeze(0).permute(0, 2, 1)
    state_iou_tensor = state_iou_tensor.unsqueeze(0).permute(0, 2, 1)
    state_entropy_tensor = state_entropy_tensor.reshape(num_regions, -1).unsqueeze(0).permute(0, 2, 1)

    selected_training_regions = []
    
    for regions_div in training_kl_div:
        #regions_div - (128)
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = policy_net(state_accuracy_tensor.to(policy_net.device), 
                                        state_iou_tensor.to(policy_net.device), 
                                        state_entropy_tensor.to(policy_net.device), 
                                        regions_div.unsqueeze(0).to(policy_net.device))
                
                topk_q_values, topk_indices = torch.topk(q_values, DatasetParams.K, dim=1)
                selected_training_regions.append(topk_indices)

        else:
            topk_indices = np.random.randint(low=0, high=num_regions, size=DatasetParams.K)
            topk_indices = torch.tensor(topk_indices).unsqueeze(0)
            selected_training_regions.append(topk_indices)

    selected_training_regions = torch.stack(selected_training_regions, dim=0).squeeze(1)

    return selected_training_regions, training_kl_div, steps_done