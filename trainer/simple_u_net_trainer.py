import os, json, time 

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from .logger import Logger

from dataset_utils.cityscapes_collate_fn import CityScapesCollateFn
from dataset_utils.cityscapes_dataset import CityScapesDataset

from model.u_net_model import UNet

from .callbacks import EarlyStopping
from dataset_utils.utils import convert_time_to_readable_format, calculate_iou, calculate_pixel_level_accuracy, plot_segmentation_map

import wandb

from dataset_utils.enums import VISUALIZATION_FILES

class SimpleUNetTrainer:

    def __init__(self, model:UNet, 
                trainer_kwargs:dict,
                optimizer_kwargs:dict,
                lr_scheduler_kwargs:dict,
                callbacks_kwargs:dict,
                dataset_kwargs:dict):
        
        # wandb.init(
        #     project="Semantic-Segmentation-U_Net-Runs",
        #     config={
        #         "dataset_kwargs":dataset_kwargs,
        #         "optimizer_kwargs":optimizer_kwargs,
        #         "trainer_kwargs":trainer_kwargs,
        #         "lr_scheduler_kwargs":lr_scheduler_kwargs,
        #         "callbacks_kwargs":callbacks_kwargs
        #     }
        # )        

        self.model = model 

        self.is_training = trainer_kwargs["is_training"]
        self.first_val_epoch = trainer_kwargs["first_val_epoch"]
        self.metric_eval_mode = trainer_kwargs["metric_eval_mode"]
        self.metric_average_mode = trainer_kwargs["metric_average_mode"]
        self.epochs = trainer_kwargs["epochs"]
        self.monitor_train = trainer_kwargs["monitor_train"]
        self.monitor_val = trainer_kwargs["monitor_val"]
        self.monitor_test = trainer_kwargs["monitor_test"]
        self.gradient_clipping = trainer_kwargs["gradient_clipping"]
        self.device_count = torch.cuda.device_count()
        self.mxp_training = trainer_kwargs["mxp_training"]
        self.loss_combination_strategy = trainer_kwargs["loss_combination_strategy"]
        self.val_segmentation_plot_dir = trainer_kwargs["val_segmentation_plot_dir"]

        if not os.path.exists(self.val_segmentation_plot_dir):
            os.makedirs(self.val_segmentation_plot_dir)    

        self.device = torch.device(trainer_kwargs["device"]) if torch.cuda.is_available() else torch.device("cpu")        
        self.model.to(self.device)

        self.output_dir = trainer_kwargs["output_dir"]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)        

        self.logger = Logger(trainer_kwargs)

        prev_layer_name = ""
        for name, param in self.model.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name != prev_layer_name:
                prev_layer_name = layer_name
                self.logger.log_block("{:<70} {:<30} {:<30} {:<30}".format('Name','Weight Shape','Total Parameters', 'Trainable'))
            self.logger.log_message("{:<70} {:<30} {:<30} {:<30}".format(name, str(param.data.shape), param.data.numel(), param.requires_grad))

        '''
        #TODO, load from ckpt
        '''

        self._init_semantic_segmentation_dataset(dataset_kwargs)

        self.logger.log_line()
        self.logger.log_message(f'Train Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'Train Annotations Directory: {dataset_kwargs["train_annotation_dir"]}')
        self.logger.log_new_line()
        self.logger.log_message(f'Train Original Images Directory: {dataset_kwargs["train_original_images_dir"]}')
        self.logger.log_new_line()
        self.logger.log_message(f'Training Batch Size: {dataset_kwargs["train_batch_size"]}')

        self.logger.log_line()

        self.logger.log_line()
        self.logger.log_message(f'Val Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'Val Annotations Directory: {dataset_kwargs["val_annotation_dir"]}')
        self.logger.log_new_line()
        self.logger.log_message(f'Val Original Images Directory: {dataset_kwargs["val_original_images_dir"]}')
        self.logger.log_new_line()
        self.logger.log_message(f'Validation Batch Size: {dataset_kwargs["val_batch_size"]}')

        self.logger.log_line()

        self.logger.log_message(f'Test Dataloader:')
        self.logger.log_new_line()
        self.logger.log_message(f'Test Annotations Directory: {dataset_kwargs["test_annotation_dir"]}')
        self.logger.log_new_line()
        self.logger.log_message(f'Test Original Images Directory: {dataset_kwargs["test_original_images_dir"]}')
        self.logger.log_new_line()
        self.logger.log_message(f'Testing Batch Size: {dataset_kwargs["test_batch_size"]}')
        
        self.num_training_steps = self.total_train_batch*self.epochs

        ''' 
        #TODO, 
        - self.num_warmup_steps = lr_scheduler_kwargs()
        '''

        self._init_optimizer(optimizer_kwargs, trainer_kwargs["load_from_checkpoint"])
        self.logger.log_line()
        self.logger.log_message(f'Optimizer: {self.optimizer.__class__.__name__}')
        self.logger.log_new_line()        

        for param_group in self.optimizer.param_groups:
            self.logger.log_message(f'model_name: {param_group["model_name"]}')
            for k,v in param_group.items():
                if k!="model_name" and k!="params":
                    self.logger.log_message("{:<30} {}".format(k, v))
            self.logger.log_new_line()                    

        ''' 
        #TODO, self._init_lr_scheduler(lr_scheduler_kwargs)
        Log lr_scheduler args: save_ckpt, patience, threshold, mode
        '''

        # put model to device
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        self._init_lr_scheduler(lr_scheduler_kwargs)

        self.logger.log_line()
        self.logger.log_message(f'Device: {self.model.device} and Device Count: {self.device_count}')
        self.logger.log_new_line()

    def _init_semantic_segmentation_dataset(self, dataset_kwargs:dict):
        def init_dataloader_helper(annotations_dir:str, original_images_dir:str, batch_size:int, dataset_type:str, image_resize:int, interpolation_strategy:str):

            cityscapes_dataset = CityScapesDataset(annotations_dir, original_images_dir, dataset_type)
            cityscapes_collate_fn = CityScapesCollateFn(image_resize=image_resize, interpolation_strategy=interpolation_strategy, split=dataset_type)

            dataloader = DataLoader(
                cityscapes_dataset, batch_size=batch_size, collate_fn=cityscapes_collate_fn
            )

            return dataloader
        
        self.train_dataloader = init_dataloader_helper(
            dataset_kwargs["train_annotation_dir"],
            dataset_kwargs["train_original_images_dir"],
            dataset_kwargs["train_batch_size"],
            dataset_type="train",
            image_resize=dataset_kwargs["image_resize"],
            interpolation_strategy=dataset_kwargs["interpolation"]
        )

        self.train_batch_size = dataset_kwargs["train_batch_size"]

        self.num_classes = len(self.train_dataloader.collate_fn.valid_classes)

        self.val_dataloader = init_dataloader_helper(
            dataset_kwargs["val_annotation_dir"],
            dataset_kwargs["val_original_images_dir"],
            dataset_kwargs["val_batch_size"],
            dataset_type="val",
            image_resize=dataset_kwargs["image_resize"],
            interpolation_strategy=dataset_kwargs["interpolation"]            
        )

        self.val_batch_size = dataset_kwargs["val_batch_size"]

        self.test_dataloader = init_dataloader_helper(
            dataset_kwargs["test_annotation_dir"],
            dataset_kwargs["test_original_images_dir"],
            dataset_kwargs["test_batch_size"],
            dataset_type="test",
            image_resize=dataset_kwargs["image_resize"],
            interpolation_strategy=dataset_kwargs["interpolation"]            
        )

        self.test_batch_size = dataset_kwargs["test_batch_size"]
        
        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10 

        self.total_validation_batch = len(self.val_dataloader)
        self.total_testing_batch = len(self.test_dataloader)           

    def _init_callbacks(self, callbacks_kwargs:dict):
        self.callbacks = EarlyStopping(self.logger, self.output_dir, **callbacks_kwargs["kwargs"])    

    def _init_optimizer(self, optimizer_kwargs:dict, load_from_checkpoint:bool):
        param_dict = []

        param_dict.append({
            "params":self.model.encoder_module.parameters(), "lr":optimizer_kwargs["encoder_lr"], "model_name":"UNet Decoder"
        })

        param_dict.append({
            "params":self.model.decoder_module.parameters(), "lr":optimizer_kwargs["encoder_lr"], "model_name":"UNet Decoder"
        })

        param_dict.append({
            "params":self.model.final_classification_layer.parameters(), "lr":optimizer_kwargs["classification_lr"], "model_name":"UNet Classifier"
        })

        self.optimizer = getattr(
            torch.optim, optimizer_kwargs["type"]
        )(param_dict, **optimizer_kwargs["kwargs"])

    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict):

        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train(self):

        self.logger.log_line()
        self.logger.log_message(f'Start Training: Max Epoch {self.epochs}')
        self.logger.log_new_line()

        self.total_training_time = 0.0

        try:
            for epoch in range(self.epochs):
                self.cur_epoch = epoch
                self.logger.log_line()

                if self.monitor_train:
                    self.train_one_epoch()

                if self.monitor_val:
                    self.valid_one_epoch()

                if self.monitor_test:
                    self.test_one_epoch()

        except KeyboardInterrupt:
            self.callbacks.exit_training(self.model)
            self.logger.log_line()
            self.logger.log_message(f'Exiting Training due to Keyboard Interrupt')
            # wandb.finish()
            exit(1)

    def train_one_epoch(self):

        self.model.train()
        total_loss = 0.0 
        ten_percent_batch_total_loss = 0

        total_mean_iou = 0.0 
        ten_percent_batch_total_mean_iou = 0.0

        total_pixel_acc = 0.0 
        ten_percent_batch_total_pixel_acc = 0.0
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0

        for batch_idx, data_items in enumerate(self.train_dataloader):
            for k,v in data_items.items():
                if torch.is_tensor(v):                    
                    data_items[k] = v.to(self.device)

            step_begin_time = time.time()
            loss, mean_iou, pixel_acc = self.train_one_step(data_items)
            step_end_time = time.time()

            total_loss += loss
            ten_percent_batch_total_loss += loss

            total_mean_iou += mean_iou
            ten_percent_batch_total_mean_iou += mean_iou

            total_pixel_acc += pixel_acc
            ten_percent_batch_total_pixel_acc += pixel_acc

            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)

            layer_name_lrs = {param_group["model_name"]: param_group["lr"] for param_group in self.optimizer.param_groups}
            log_lrs = "" # log lr of each layers
            for layer_name, lr in layer_name_lrs.items():
                log_lrs += f" - {layer_name} lr: {lr:.2e}"

            if self.total_train_batch < 10:
                msg = f'Epoch: {self.cur_epoch} - iteration {batch_idx}/{self.total_train_batch} - total loss {total_loss:.4f}'
                self.logger.log_message(message=msg)            

            elif (batch_idx + 1) % self.ten_percent_train_batch == 0:
                average_loss = ten_percent_batch_total_loss/self.ten_percent_train_batch
                average_time = ten_percent_training_time/self.ten_percent_train_batch

                average_mean_iou = ten_percent_batch_total_mean_iou/self.ten_percent_train_batch
                average_pixel_acc = ten_percent_batch_total_pixel_acc/self.ten_percent_train_batch

                sec_per_batch_log_message = f" - secs/batch {convert_time_to_readable_format(round(average_time, 4))}"
                message = f"Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - total loss {average_loss:.4f} - total mean iou {average_mean_iou:.4f} - total pixel acc {average_pixel_acc:.4f}" + log_lrs + sec_per_batch_log_message
                self.logger.log_message(message=message)

                ten_percent_batch_total_loss = 0
                ten_percent_training_time = 0 

                ten_percent_batch_total_mean_iou = 0
                ten_percent_batch_total_pixel_acc = 0

        self.total_training_time += epoch_training_time
        self.logger.log_message(f"Epoch #{self.cur_epoch}: Average Loss {total_loss/self.total_train_batch} - Epoch Training Time: {convert_time_to_readable_format(round(epoch_training_time, 4))} - Total Training Time: {convert_time_to_readable_format(round(epoch_training_time, 4))}")

    def train_one_step(self, data_items):

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            loss, predicted_segmentation_map, label_tensors = self.model(**data_items)
            loss.backward()

            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

            self.optimizer.step()

        self.lr_scheduler.step(epoch=self.cur_epoch)

        ''' 
        #TODO, self.lr_scheduler.step()
        '''

        _, mean_iou = calculate_iou(predicted_segmentation_map, label_tensors, self.num_classes)
        pixel_acc = calculate_pixel_level_accuracy(predicted_segmentation_map, label_tensors)
        
        return loss.item(), mean_iou.item(), pixel_acc.item()
    
    def valid_one_epoch(self):

        self.model.eval()

        total_valid_loss = 0.0 
        total_mean_iou = 0.0 
        total_pixel_acc = 0.0 
        
        epoch_validation_time = 0.0

        with torch.no_grad():
            for batch_idx, data_items in enumerate(self.val_dataloader):
                for k,v in data_items.items():
                    if torch.is_tensor(v):                    
                        data_items[k] = v.to(self.device)

                original_image_paths = data_items["original_image_paths"]
                gtfine_image_paths = data_items["gtfine_image_paths"]

                del data_items["original_image_paths"]
                del data_items["gtfine_image_paths"]

                step_begin_time = time.time()
                loss, mean_iou, pixel_acc, predicted_segmentation_map = self.valid_one_step(data_items)             

                if batch_idx == 0:
                    
                    plot_segmentation_map(
                        predicted_segmentation_map, data_items["label_tensors"], gtfine_image_paths, self.val_dataloader.collate_fn.category_id_2_color, self.val_segmentation_plot_dir
                    )
                
                step_end_time = time.time()

                total_valid_loss += loss 
                total_mean_iou += mean_iou
                total_pixel_acc += pixel_acc

                epoch_validation_time += (step_end_time - step_begin_time)

        avg_valid_loss =  total_valid_loss/self.total_validation_batch
        avg_valid_pixel_acc = total_pixel_acc/self.total_validation_batch
        avg_valid_mean_iou = total_mean_iou/self.total_validation_batch

        self.logger.log_line()
        self.logger.log_message(
            f'Epoch #{self.cur_epoch}: Average Validation Loss: {avg_valid_loss:.4f} - Average Validation Pixel Acc: {avg_valid_pixel_acc:.4f} - Average Validation Mean IOU: {avg_valid_mean_iou:.4f}'
        )
        self.logger.log_new_line()

        torch.cuda.empty_cache()

    def valid_one_step(self, data_items):
        loss, predicted_segmentation_map, label_tensors = self.model(**data_items)
        _, mean_iou = calculate_iou(predicted_segmentation_map, label_tensors, self.num_classes)
        pixel_acc = calculate_pixel_level_accuracy(predicted_segmentation_map, label_tensors)

        label_tensors = data_items["label_tensors"]
        
        batch_size, img_h, img_w  = label_tensors.shape
        
        predicted_segmentation_map = torch.argmax(predicted_segmentation_map, dim=1)                    
        predicted_segmentation_map = predicted_segmentation_map.reshape(batch_size, img_h, img_w)

        return loss.item(), mean_iou.item(), pixel_acc.item(), predicted_segmentation_map        

    def test_one_epoch(self):

        self.model.eval()

        total_test_loss = 0.0 
        total_mean_iou = 0.0 
        total_pixel_acc = 0.0 
        
        epoch_validation_time = 0.0

        with torch.no_grad():
            for batch_idx, data_items in enumerate(self.test_dataloader):
                for k,v in data_items.items():
                    if torch.is_tensor(v):                    
                        data_items[k] = v.to(self.device)

                original_image_paths = data_items["original_image_paths"]
                gtfine_image_paths = data_items["gtfine_image_paths"]

                del data_items["original_image_paths"]
                del data_items["gtfine_image_paths"]

                step_begin_time = time.time()
                loss, mean_iou, pixel_acc = self.test_one_step(data_items)
                step_end_time = time.time()

                total_test_loss += loss 
                total_mean_iou += mean_iou
                total_pixel_acc += pixel_acc

                epoch_validation_time += (step_end_time - step_begin_time)

                print(f'Test Loss: {loss} - Mean IOU: {mean_iou} - Pixel Acc: {pixel_acc}')

        avg_test_loss =  total_test_loss/self.total_testing_batch
        avg_test_pixel_acc = total_pixel_acc/self.total_testing_batch
        avg_test_mean_iou = total_mean_iou/self.total_testing_batch

        self.logger.log_line()
        self.logger.log_message(
            f'Epoch #{self.cur_epoch}: Average Testing Loss: {avg_test_loss:.4f} - Average Validation Pixel Acc: {avg_test_pixel_acc:.4f} - Average Validation Mean IOU: {avg_test_mean_iou:.4f}'
        )
        self.logger.log_new_line()

    def test_one_step(self, data_items):
        loss, predicted_segmentation_map, label_tensors = self.model(**data_items)
        _, mean_iou = calculate_iou(predicted_segmentation_map, label_tensors, self.num_classes)
        pixel_acc = calculate_pixel_level_accuracy(predicted_segmentation_map, label_tensors)
        
        return loss.item(), mean_iou.item(), pixel_acc.item()        

