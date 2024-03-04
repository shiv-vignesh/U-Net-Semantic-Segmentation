from model.u_net_model import UNet

from dataset_utils.cityscapes_collate_fn import CityScapesCollateFn
from trainer.simple_u_net_trainer import SimpleUNetTrainer

import torch
import json

def create_model(model_kwargs:dict, trainer_kwargs:dict):

    device = torch.device(trainer_kwargs["device"]) if torch.cuda.is_available() else torch.device("cpu")


    if model_kwargs["model_arch"] == "simple_u_net":
        model = UNet(output_classes=len(CityScapesCollateFn().valid_classes), device=device)
        
    model.to(device)

    return model

if __name__ == "__main__":

    config = json.load(open('config.json'))

    model = create_model(config["model_kwargs"], config["trainer_kwargs"])

    trainer = SimpleUNetTrainer(model, 
                                config["trainer_kwargs"], config["optimizer_kwargs"], 
                                config["lr_scheduler_kwargs"], config["callbacks_kwargs"], 
                                config["dataset_kwargs"]
                                )
    
    trainer.train()
