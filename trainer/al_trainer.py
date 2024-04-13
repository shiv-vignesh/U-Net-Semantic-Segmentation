import os, json, time 
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler

from .logger import Logger

from dataset_utils.al_dataset import ActiveLearningDataset
from dataset_utils.al_collate_fn import ActiveLearningCollateFn
from dataset_utils.enums import DatasetParams, RLParams

from model.u_net_model import UNet
from model.dqn import DQN, ExperienceReplay, Transition, State
from model.model_utils import calculate_crop_region_accuracy, calculate_crop_region_iou, calculate_crop_region_entropy, calculate_label_distribution, select_action, compute_reward_weights

from .callbacks import EarlyStopping
from dataset_utils.utils import convert_time_to_readable_format, calculate_iou, calculate_pixel_level_accuracy, calculate_iou_for_classes, calculate_accuracy_for_classes

import wandb

from dataset_utils.enums import VISUALIZATION_FILES

class ActiveLearningTrainer:

    def __init__(self, model:UNet, 
                policy_net:DQN, 
                target_net:DQN,
                trainer_kwargs:dict,
                optimizer_kwargs:dict,
                lr_scheduler_kwargs:dict,
                callbacks_kwargs:dict,
                dataset_kwargs:dict):
        
        self.model = model
        
        self.policy_net = policy_net
        self.target_net = target_net
        
        self.output_dir = trainer_kwargs["output_dir"]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)                

        self.logger = Logger(trainer_kwargs)
        self.minority_class_indices = trainer_kwargs["minority_classes"]

        self.logger.log_line()
        self.logger.log_message(f'U-Net Model device: {self.model.device}')
        self.logger.log_new_line()
        self.logger.log_message(f'Policy device: {self.policy_net.device}; Target Device: {self.target_net.device}')
        
        self.num_episodes = trainer_kwargs["num_episodes"]

        ''' 
        change to U-net.device & DQN.device
        '''

        self.u_net_device = self.model.device
        self.dqn_device = self.policy_net.device

        self._init_active_learning_dataset(dataset_kwargs)
        ''' 
        TODO; write code to train only encoder or decoder, by adding only those params to optimizer.  
        '''
        self._init_optimizer(optimizer_kwargs, trainer_kwargs["load_from_checkpoint"])
        self.logger.log_line()
        self.logger.log_message(f'Optimizer: {self.optimizer.__class__.__name__}')
        self.logger.log_new_line()           

        self.gradient_clipping = trainer_kwargs["gradient_clipping"]

        self._init_lr_scheduler(lr_scheduler_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'Optimizer: {self.lr_scheduler.__class__.__name__}')
        self.logger.log_new_line()     

        self._init_callbacks(callbacks_kwargs)
        
        self.experience_replay = ExperienceReplay()
        self.policy_net_batch_size = trainer_kwargs["policy_net_batch_size"]

    def _init_active_learning_dataset(self, dataset_kwargs:dict):
        def init_dataloader_helper(annotations_dir:str, original_images_dir:str, batch_size:int, dataset_type:str, image_resize:int, interpolation_strategy:str):

            cityscapes_dataset = ActiveLearningDataset(annotations_dir, original_images_dir, dataset_type)
            cityscapes_collate_fn = ActiveLearningCollateFn(image_resize=image_resize, interpolation_strategy=interpolation_strategy, split=dataset_type)

            dataloader = DataLoader(
                cityscapes_dataset, batch_size=batch_size, collate_fn=cityscapes_collate_fn
            )

            return dataloader
        
        self.state_dataloader = init_dataloader_helper(
            dataset_kwargs["train_annotation_dir"],
            dataset_kwargs["train_original_images_dir"],
            dataset_kwargs["state_batch_size"],
            dataset_type="state",
            image_resize=dataset_kwargs["image_resize"],
            interpolation_strategy=dataset_kwargs["interpolation"]            
        )

        self.logger.log_line()
        self.logger.log_message(f'State Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'State Batch Size: {dataset_kwargs["state_batch_size"]}')
        self.logger.log_new_line()

        self.reward_dataloader = init_dataloader_helper(
            dataset_kwargs["train_annotation_dir"],
            dataset_kwargs["train_original_images_dir"],
            dataset_kwargs["reward_batch_size"],
            dataset_type="reward",
            image_resize=dataset_kwargs["image_resize"],
            interpolation_strategy=dataset_kwargs["interpolation"]            
        )

        self.logger.log_line()
        self.logger.log_message(f'Reward Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'Reward Batch Size: {dataset_kwargs["reward_batch_size"]}')
        self.logger.log_new_line()

        self.train_dataloader = init_dataloader_helper(
            dataset_kwargs["train_annotation_dir"],
            dataset_kwargs["train_original_images_dir"],
            dataset_kwargs["train_batch_size"],
            dataset_type="train",
            image_resize=dataset_kwargs["image_resize"],
            interpolation_strategy=dataset_kwargs["interpolation"]            
        )

        self.logger.log_line()
        self.logger.log_message(f'Train Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'Train Batch Size: {dataset_kwargs["train_batch_size"]}')
        self.logger.log_new_line()

        self.validation_dataloader = init_dataloader_helper(
            dataset_kwargs["val_annotation_dir"],
            dataset_kwargs["val_original_images_dir"],
            dataset_kwargs["val_batch_size"],
            dataset_type="validation",
            image_resize=dataset_kwargs["image_resize"],
            interpolation_strategy=dataset_kwargs["interpolation"]            
        )

        self.logger.log_line()
        self.logger.log_message(f'Validation Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'Validation Batch Size: {dataset_kwargs["val_batch_size"]}')
        self.logger.log_new_line()

        self.num_classes = len(self.state_dataloader.collate_fn.valid_classes)
        self.classes = self.state_dataloader.collate_fn.valid_classes

        #Remains fixed/constant throughout. (128, num_classes)
        self.state_label_distribution = None

        if not os.path.exists(f'al_partition/training_regions_distribution.pt'):
            print(f'Computing Training Regions Distribution')
            self.compute_training_regions_distributions()
        else:
            print('Loading Training Regions Distribution')
            self.training_regions_distribution = torch.load('al_partition/training_regions_distribution.pt')
            self.training_regions_distribution = self.training_regions_distribution.to(self.u_net_device)

    def _init_callbacks(self, callbacks_kwargs:dict):
        self.callbacks = EarlyStopping(self.logger, self.output_dir, **callbacks_kwargs["kwargs"])   

    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict):
        
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.lr_scheduler_policy = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)

    def _init_optimizer(self, optimizer_kwargs:dict, load_from_checkpoint:bool):

        self.logger.log_line()
        self.logger.log_message(f'Initializing U-Net model Parameters')
        self.logger.log_new_line()

        param_dict = []

        u_net_optimizer_kwargs = optimizer_kwargs["u_net_kwargs"]

        param_dict.append({
            "params":self.model.encoder_module.parameters(), "lr":u_net_optimizer_kwargs["encoder_lr"], "model_name":"UNet Decoder"
        })

        param_dict.append({
            "params":self.model.decoder_module.parameters(), "lr":u_net_optimizer_kwargs["encoder_lr"], "model_name":"UNet Decoder"
        })

        param_dict.append({
            "params":self.model.final_classification_layer.parameters(), "lr":u_net_optimizer_kwargs["classification_lr"], "model_name":"UNet Classifier"
        })

        param_dict.append({
            "params":self.model.pre_classification_layer.parameters(), "lr":u_net_optimizer_kwargs["classification_lr"], "model_name":"UNet Classifier"
        })

        self.optimizer = getattr(
            torch.optim, u_net_optimizer_kwargs["type"]
        )(param_dict, **u_net_optimizer_kwargs["kwargs"])    

        if self.policy_net.__class__.__name__ == "DQN":
            
            param_dict = []
            dqn_optimizer_kwargs = optimizer_kwargs["dqn_kwargs"]

            if dqn_optimizer_kwargs["type"] == "RMSprop":
                del dqn_optimizer_kwargs["kwargs"]["amsgrad"]
            
            self.policy_optimizer = getattr(
                torch.optim, dqn_optimizer_kwargs["type"]                
            )(self.policy_net.parameters(), **dqn_optimizer_kwargs["kwargs"])

    def compute_training_regions_distributions(self):
                
        def region_distribution(label_tensor:torch.tensor):
            label_distribution_list = []

            for idx, crop_region in enumerate(self.train_dataloader.collate_fn.crops):
                start_x, start_y = crop_region[0]
                end_x, end_y = crop_region[1]

                label_crop = label_tensor[start_x:end_x, start_y:end_y]
                flattened_crop = label_crop.flatten()

                counts = torch.bincount(flattened_crop, minlength=len(self.classes))
                crop_distribution = counts.float()/counts.sum()      

                label_distribution_list.append(crop_distribution)

            return torch.stack(label_distribution_list, dim=0)  #(128, num_classes)                              

        self.training_regions_distribution = []

        for _, data_items in enumerate(self.train_dataloader):
            label_tensors = data_items["label_tensors"].to(self.u_net_device)

            for label_tensor in label_tensors:
                label_distribution = region_distribution(label_tensor)                
                self.training_regions_distribution.extend([label_distribution])

        #[len(training_dataset), 128, num_classes]
        self.training_regions_distribution = torch.stack(self.training_regions_distribution, dim=0)

        torch.save(self.training_regions_distribution, 'al_partition/training_regions_distribution.pt')

    def compute_current_state(self):
        
        predicted_tensor_list, label_tensor_list = [], []

        with torch.no_grad():
            for _, data_items in enumerate(self.state_dataloader):
                for k,v in data_items.items():
                    if torch.is_tensor(v):                    
                        data_items[k] = v.to(self.u_net_device)   

                _, predicted_segmentation_map, label_tensors = self.model(**data_items)             
                predicted_tensor_list.append(predicted_segmentation_map.squeeze(0))
                label_tensor_list.append(label_tensors.squeeze(0))

        #(10, n_c, 512, 1024)
        predicted_tensor_list = torch.stack(predicted_tensor_list, dim=0)
        #(10, 512, 1024)
        label_tensor_list = torch.stack(label_tensor_list, dim=0)

        state_accuracy = []
        state_mean_iou = []
        state_entropy = []

        for idx, crop_region in enumerate(self.state_dataloader.collate_fn.crops):
            ''' 
            Computes metrics over all the predictions. (full state dataset=10)
            '''
            
            start_x, start_y = crop_region[0]
            end_x, end_y = crop_region[1]

            prediction_crop = predicted_tensor_list[:, :, start_x:end_x, start_y:end_y]
            label_crop = label_tensor_list[:, start_x:end_x, start_y:end_y]

            accuracy_per_class = calculate_crop_region_accuracy(prediction_crop, label_crop, self.classes) # tensorSize(8)
            mean_iou_per_class = calculate_crop_region_iou(prediction_crop, label_crop, self.classes) # tensorSize(8)
            entropy = calculate_crop_region_entropy(prediction_crop) # tensorSize(64, 64)

            state_accuracy.append(accuracy_per_class)
            state_mean_iou.append(mean_iou_per_class)
            state_entropy.append(entropy)

        if self.state_label_distribution is None:
            self.state_label_distribution = []

            for idx, crop_region in enumerate(self.state_dataloader.collate_fn.crops):
                start_x, start_y = crop_region[0]
                end_x, end_y = crop_region[1]

                label_crop = label_tensor_list[:, start_x:end_x, start_y:end_y]
                distribution = calculate_label_distribution(label_crop, self.classes)

                self.state_label_distribution.append(distribution)

            #(128, num_classes); average across all samples of state_dataset
            self.state_label_distribution = torch.stack(self.state_label_distribution, dim=0)               

        state_accuracy = torch.stack(state_accuracy, dim=0)
        state_mean_iou = torch.stack(state_mean_iou, dim=0)
        state_entropy = torch.stack(state_entropy, dim=0)
        
        return state_accuracy, state_mean_iou, state_entropy
    
    def compute_reward(self, episode:int, cur_step:int):

        with torch.no_grad():            
            reward_loss, overall_reward_acc, overall_reward_iou = 0.0, 0.0, 0.0

            reward_per_class_iou = torch.zeros(self.num_classes)
            reward_per_class_acc = torch.zeros(self.num_classes)

            for batch_idx, data_items in enumerate(self.reward_dataloader):
                for k,v in data_items.items():
                    if torch.is_tensor(v):                    
                        data_items[k] = v.to(self.u_net_device)

                loss, predicted_segmentation_map, label_tensors = self.model(**data_items)
                iou_per_class, mean_iou = calculate_iou(predicted_segmentation_map, label_tensors, self.num_classes)
                accuracy_per_class, total_accuracy = calculate_pixel_level_accuracy(predicted_segmentation_map, label_tensors, self.num_classes)

                reward_per_class_acc += accuracy_per_class
                reward_per_class_iou += iou_per_class

                reward_loss += loss.item() 
                overall_reward_iou += mean_iou.item()
                overall_reward_acc += total_accuracy.item()

            reward_loss =  reward_loss/len(self.reward_dataloader)
            overall_reward_acc = overall_reward_acc/len(self.reward_dataloader)
            overall_reward_iou = overall_reward_iou/len(self.reward_dataloader)

            reward_per_class_iou = reward_per_class_iou/len(self.reward_dataloader)
            reward_per_class_acc = reward_per_class_acc/len(self.reward_dataloader)

        self.logger.log_line()
        self.logger.log_message(f'Episode: {episode} Current Step: {cur_step}; Average Reward Loss: {reward_loss:.4f} Average Pixel Acc: {overall_reward_acc:.4f} Average IOU: {overall_reward_iou:.4f}')
        self.logger.log_message(f'Reward Dataset Accuracy Per Class: {reward_per_class_acc} IOU Per class: {reward_per_class_iou}')
        self.logger.log_new_line()        

        return reward_loss, overall_reward_acc, overall_reward_iou, reward_per_class_acc, reward_per_class_iou


    def train(self):
        ''' 
        2-stage Q-Learning. Sample and Region. 
        '''

        self.steps_done = 0
        self.cur_episode = 0
        self.cur_actual_train_epoch = 0

        for episode in range(self.num_episodes):
            self.logger.log_line()
            self.logger.log_message(f'Computing state representation for episode: {episode}')
            self.logger.log_new_line()

            self.cur_episode = episode
            regions_covered = 0
            episode_steps = 0
            total_al_training_loss = 0

            budget_reached = False

            
            '''TODO-Calculate Per class/Category Metric'''
            state_accuracy, state_mean_iou, state_entropy = self.compute_current_state()

            '''
            Reward only calculated for underrepresented classes
            '''
            reward_loss, reward_acc, reward_iou, reward_per_class_acc, reward_per_class_iou = self.compute_reward(self.cur_episode, episode_steps)
            # reward_weights = compute_reward_weights(reward_per_class_acc, reward_per_class_iou, self.minority_class_indices)

            ''' 
            TODO-unique regions covered per training image. 
            '''

            #at every step, the regions covered is (150 * 8)
            episode_loss, episode_reward = 0.0, 0.0
            while not budget_reached:
                selected_training_regions, training_kl_div, steps_done = select_action(
                    state_label_distribution=self.state_label_distribution,
                    training_label_distribution=self.training_regions_distribution,
                    state_accuracy_tensor=state_accuracy,
                    state_iou_tensor=state_mean_iou, 
                    state_entropy_tensor=state_entropy, 
                    steps_done=self.steps_done,
                    policy_net=self.policy_net
                )

                current_states = []

                for kl_div in training_kl_div:
                    current_state = State(
                        state_accuracy.unsqueeze(0).permute(0, 2, 1), 
                        state_mean_iou.unsqueeze(0).permute(0, 2, 1), 
                        state_entropy.reshape(state_entropy.size(0), -1).unsqueeze(0).permute(0, 2, 1),
                        kl_div.unsqueeze(0)
                        )
                    
                    current_states.append(current_state)
                                
                al_training_loss = self.train_u_net(selected_training_regions, episode_steps)
                al_training_loss = 0.0
                total_al_training_loss += al_training_loss

                episode_steps += 1
                self.steps_done += 1
                regions_covered += selected_training_regions.flatten().size(0)

                next_reward_loss, next_reward_acc, next_reward_iou, next_reward_per_class_acc, next_reward_per_class_iou = self.compute_reward(self.cur_episode, episode_steps)

                # delta_reward_iou = next_reward_per_class_iou[self.minority_class_indices] - reward_per_class_iou[self.minority_class_indices]
                # delta_reward_acc = next_reward_per_class_acc[self.minority_class_indices] - reward_per_class_acc[self.minority_class_indices]
                # reward = (reward_weights * delta_reward_acc).sum() + (reward_weights * delta_reward_iou).sum()

                delta_reward_iou = next_reward_per_class_iou - reward_per_class_iou
                delta_reward_acc = next_reward_per_class_acc - reward_per_class_acc
                reward = (delta_reward_acc).sum() + (delta_reward_iou).sum()

                episode_reward += reward.item()

                if regions_covered <= DatasetParams.BUDGET:
                    #next state after training U-net
                    '''
                    TODO- Reassign to current_state
                    '''
                    next_state_accuracy, next_state_mean_iou, next_state_entropy = self.compute_current_state()
                    next_states = []

                    for kl_div in training_kl_div:
                        next_state = State(
                            next_state_accuracy.unsqueeze(0).permute(0, 2, 1), 
                            next_state_mean_iou.unsqueeze(0).permute(0, 2, 1), 
                            next_state_entropy.reshape(state_entropy.size(0), -1).unsqueeze(0).permute(0, 2, 1),
                            kl_div.unsqueeze(0)
                            )
                        next_states.append(next_state)

                    self.experience_replay.add_memory(current_states, selected_training_regions, reward, next_states)

                    reward_per_class_iou = next_reward_per_class_iou
                    reward_per_class_acc = next_reward_per_class_acc

                    # reward_weights = compute_reward_weights(reward_per_class_acc, reward_per_class_iou, self.minority_class_indices)
                    state_accuracy = next_state_accuracy
                    state_mean_iou = next_state_mean_iou
                    state_entropy = next_state_mean_iou

                else:
                    budget_reached = True
                    next_states = None 

                    self.experience_replay.add_memory(
                        current_states, selected_training_regions, reward, next_states
                    )

                print(f'Experience Replay Length: {len(self.experience_replay)}')

                if len(self.experience_replay) < self.policy_net_batch_size:
                    continue

                else:
                    loss = self.train_policy_network()
                    print(f'Loss: {loss} Reward: {reward} Delta Acc: {delta_reward_acc} Delta IOU: {delta_reward_iou}')
                    episode_loss += loss.item()

                target_net_dict = self.target_net.state_dict()
                policy_net_dict = self.policy_net.state_dict()

                for key in target_net_dict:
                    target_net_dict[key] = RLParams.TAU * target_net_dict[key] + (1-RLParams.TAU) * policy_net_dict[key]
                
                self.target_net.load_state_dict(target_net_dict)

                del target_net_dict
                del policy_net_dict
            
    def train_u_net(self, selected_training_regions:torch.tensor, episode_step:int, batch_size:int=2):
        ''' 
        selected_training_regions - (150, 8)
        '''

        training_corpus = {
            "image_tensors":[],
            "label_tensors":[],
            "region_masks":[]
        }
        ''' 
        train_dataloader will have bs_size = 1. Easier to create mask tensors for each image in DT.
        '''
        
        num_samples = list(range(selected_training_regions.shape[0]))
        random.shuffle(num_samples)

        for i, data_items in enumerate(self.train_dataloader):
            for k,v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.u_net_device)

            bs, h, w = data_items["image_tensors"].shape[0], data_items["image_tensors"].shape[-2], data_items["image_tensors"].shape[-1]
            mask = torch.zeros(size=(bs, h, w)).to(self.u_net_device)

            selected_regions = selected_training_regions[i]

            for idx, crop_region in enumerate(self.train_dataloader.collate_fn.crops):
                if idx in selected_regions:
                    start_x, start_y = crop_region[0]
                    end_x, end_y = crop_region[1]                

                    mask[:,start_x:end_x, start_y:end_y] = 1

            training_corpus["image_tensors"].append(data_items["image_tensors"])            
            training_corpus["label_tensors"].append(data_items["label_tensors"])            
            training_corpus["region_masks"].append(mask)            
        
        for k, v in training_corpus.items():
            training_corpus[k] = torch.concat(v, dim=0).to(self.u_net_device)

        dataset = TensorDataset(training_corpus["image_tensors"], training_corpus["label_tensors"], training_corpus["region_masks"])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        al_training_loss = 0.0

        for batch_images, batch_labels, batch_masks in dataloader:
            data_items = {
                "image_tensors":batch_images,
                "label_tensors":batch_labels,
                "region_masks":batch_masks.float()
            }

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                loss, _, _ = self.model(**data_items)
                loss.backward()

                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

                al_training_loss += loss
                
            self.optimizer.step()
            self.lr_scheduler.step()
        
        ''' 
        compute iou & acc for train_dataset
        '''
        
        self.logger.log_new_line()
        self.logger.log_message(f'Active Learning episode: {self.cur_episode}; step:{episode_step}; Loss: {al_training_loss/len(dataloader)}')

        return al_training_loss


    def train_policy_network(self):
        
        transitions = self.experience_replay.sample(self.policy_net_batch_size)

        curren_states = [t.state for t in transitions]
        rewards = torch.tensor([t.reward for t in transitions]).unsqueeze(1).to(self.dqn_device)
        selected_regions = [t.action for t in transitions]

        curren_q_values = torch.zeros((len(transitions), DatasetParams.K), device=self.dqn_device)

        self.policy_optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            for idx, state in enumerate(curren_states):
                q_values = self.policy_net(
                    state.state_dataset_acc.to(self.dqn_device),
                    state.state_dataset_iou.to(self.dqn_device),
                    state.state_dataset_entropy.to(self.dqn_device),
                    state.training_image_kl_div.to(self.dqn_device)
                )

                selected_region = selected_regions[idx].unsqueeze(0)
                selected_q_values = torch.gather(q_values, dim=1, index=selected_region.to(self.dqn_device))

                curren_q_values[idx] = selected_q_values.squeeze(0)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, [t.next_state for t in transitions if t.next_state is not None])))
        non_final_next_states =[t.next_state for t in transitions if t.next_state is not None]

        next_state_values = torch.zeros((len(transitions), DatasetParams.K), device=self.dqn_device)
        
        with torch.no_grad():
            for idx, next_state in enumerate(non_final_next_states):
                q_values = self.target_net(
                    next_state.state_dataset_acc.to(self.dqn_device),
                    next_state.state_dataset_iou.to(self.dqn_device),
                    next_state.state_dataset_entropy.to(self.dqn_device),
                    next_state.training_image_kl_div.to(self.dqn_device)
                )
                topk_q_values, topk_indices = torch.topk(q_values, DatasetParams.K, dim=1)
                next_state_values[idx] = topk_q_values.squeeze(0)
                
        expected_q_values = (next_state_values * RLParams.GAMMA) + rewards

        loss = torch.nn.SmoothL1Loss()(curren_q_values, expected_q_values)
        loss.backward()

        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clipping)

        self.policy_optimizer.step()
        self.lr_scheduler_policy.step()

        return loss