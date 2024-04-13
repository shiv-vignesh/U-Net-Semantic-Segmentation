import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import deque

from dataset_utils.enums import RLParams

class State:
    def __init__(self, state_dataset_acc:torch.tensor, state_dataset_iou:torch.tensor, state_dataset_entropy:torch.tensor, training_image_kl_div:torch.tensor):
        ''' 
        state_dataset_acc - (1, n_c, 128
        state_dataset_iou - (1, n_c, 128)
        state_dataset_entropy - (1, 128, 4096)
        training_kl_div - (1, 128)
        '''

        self.state_dataset_acc = state_dataset_acc
        self.state_dataset_iou = state_dataset_iou
        self.state_dataset_entropy = state_dataset_entropy
        self.training_image_kl_div = training_image_kl_div

class Transition:
  def __init__(self, state: State, action:torch.tensor, reward:torch.tensor, next_state:State):
    
    '''Delayed Reward. Same reward for all the training image kl_div'''
    '''Action is the indices of selected regions by policy net. (150, topk_idxes)'''
    '''Use these indices to identify the necessary Q-values over which to compute TD error'''

    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state

class ExperienceReplay:
    def __init__(self):
        self.memory_buffer = deque([], maxlen=RLParams.EXP_REPLAY_CAPACITY)

    def add_memory(self, current_states, selected_training_regions, reward, next_states):        
        if next_states:
            for idx, (current_state, next_state) in enumerate(zip(current_states, next_states)):
                selected_regions = selected_training_regions[idx]
                transition = Transition(current_state, selected_regions, reward, next_state)
                self.memory_buffer.append(transition)
        else:
            for idx, current_state in enumerate(current_states):
                selected_regions = selected_training_regions[idx]
                transition = Transition(current_state, selected_regions, reward, None)
                self.memory_buffer.append(transition)


    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
        return [self.memory_buffer[idx] for idx in indices]

    def __len__(self):
        return len(self.memory_buffer) 


class ConvBlock(nn.Module):

    def __init__(self, input_channels:int, output_channels:int, kernal_size:int=3):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernal_size, padding=(kernal_size-1)//2)
        self.batch_norm = nn.BatchNorm1d(num_features=output_channels)

    def forward(self, input_tensor:torch.tensor):

        x = self.conv(input_tensor)
        x = self.batch_norm(x)
        return F.relu(x)
    
class DQN(nn.Module):

    def __init__(self, num_classes:int, flatten_entropy:int=4096, output_size:int=1, device="cuda"):
        super(DQN, self).__init__()

        ''' 
        Increase dimensions (next enhancement)
        '''

        self.acc_conv_block = nn.Sequential(
            ConvBlock(num_classes, 32), 
            ConvBlock(32, 64)
        )

        self.iou_conv_block = nn.Sequential(
            ConvBlock(num_classes, 32), 
            ConvBlock(32, 64)
        )

        self.entropy_conv_block = ConvBlock(flatten_entropy, flatten_entropy)

        self.acc_1x1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1), 
            nn.BatchNorm1d(num_features=1)
        )
        self.iou_1x1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1),
            nn.BatchNorm1d(num_features=1)
        )
        self.entropy_1x1 = nn.Sequential(
            nn.Conv1d(in_channels=flatten_entropy, out_channels=1, kernel_size=1), 
            nn.BatchNorm1d(num_features=1)
        )

        self.concatenated_feature_block = nn.Sequential(
            ConvBlock(4, 32),
            ConvBlock(32, 64)
        )

        self.final_layer = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1) #predict a Q-Value for each region. 

        self.device = device

    def forward(self, state_accuracy_tensor:torch.tensor, state_iou_tensor:torch.tensor, 
                state_entropy_tensor:torch.tensor, regions_divergence_tensor:torch.tensor):
        
        acc_output = self.acc_conv_block(state_accuracy_tensor)
        iou_output = self.iou_conv_block(state_iou_tensor)
        entropy_output = self.entropy_conv_block(state_entropy_tensor)

        acc_output = F.relu(self.acc_1x1(acc_output)) #(bs, 1, 128)
        iou_output = F.relu(self.iou_1x1(iou_output)) #(bs, 1, 128)
        entropy_output = F.relu(self.entropy_1x1(entropy_output)) #(bs, 1, 128)

        concatenated_features = torch.concat([acc_output, iou_output, entropy_output, regions_divergence_tensor.unsqueeze(1)], dim=1)
        output = self.concatenated_feature_block(concatenated_features)
        q_values = self.final_layer(output).squeeze(1)

        return q_values

if __name__ == "__main__":

    state_accuracy_tensor = torch.randn(128, 8).unsqueeze(0).permute(0, 2, 1)
    state_iou_tensor = torch.randn(128, 8).unsqueeze(0).permute(0, 2, 1)
    state_entropy_tensor = torch.randn(128, 4096).unsqueeze(0).permute(0, 2, 1)

    regions_divergence_tensor = torch.randn(1, 128)

    dqn = DQN(
        num_classes=8
    )

    dqn(
        state_accuracy_tensor, state_iou_tensor, state_entropy_tensor, regions_divergence_tensor
    )

    


