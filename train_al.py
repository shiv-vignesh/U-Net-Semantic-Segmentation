import os
import torch
import json

from model.u_net_model import UNet
from model.dqn import DQN
from dataset_utils.cityscapes_collate_fn import CityScapesCollateFn
from trainer.al_trainer import ActiveLearningTrainer

def create_segmentation_model(model_kwargs:dict, trainer_kwargs:dict):

    device = torch.device(trainer_kwargs["device"]) if torch.cuda.is_available() else torch.device("cpu")

    if model_kwargs["model_arch"] == "simple_u_net":
        model = UNet(output_classes=len(CityScapesCollateFn().valid_classes), device=device)
        
    model.to(device)

    return model

def initialize_models(segmentation_model_path:str, device_ids:list):
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else torch.device("cpu")
    model = UNet(output_classes=len(CityScapesCollateFn().valid_classes), device=device)

    if os.path.exists(segmentation_model_path):
        model.load_state_dict(
            torch.load(segmentation_model_path)
        )
        
    model.to(device)

    device = torch.device(f"cuda:{device_ids[1]}") if torch.cuda.is_available() else torch.device("cpu")
    
    policy_net = DQN(len(CityScapesCollateFn().valid_classes), device=device)
    target_net = DQN(len(CityScapesCollateFn().valid_classes), device=device)

    policy_net.to(device)
    target_net.to(device)

    return model, policy_net, target_net

if __name__ == "__main__":

    config = json.load(open('al_config.json'))

    model, policy_net, target_net = initialize_models('residual_attention_u_net_batch_size=2/model_checkpoints/best-model.pt', device_ids=[3, 4] )

    trainer = ActiveLearningTrainer(model, policy_net, target_net,
                                config["trainer_kwargs"], config["optimizer_kwargs"], 
                                config["lr_scheduler_kwargs"], config["callbacks_kwargs"], 
                                config["dataset_kwargs"])
    
    trainer.train()