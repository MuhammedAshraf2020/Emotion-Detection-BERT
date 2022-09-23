import os
import torch
import hydra

from data  import Data
from model import Model

import pytorch_lightning as pl

@hydra.main(config_path = "./configs" , config_name = "config")
def from_torch_to_onnx(cfg):
    model_path = os.path.join(hydra.utils.get_original_cwd() , "models" , "best-checkpoint.ckpt")
    # define the model and load the checkpoint
    emotion_model = Model.load_from_checkpoint(model_path , cfg = cfg)
    # we need to preapre and setup the date here
    emotion_data  = Data(cfg)
    emotion_data.prepare_data()
    emotion_data.setup()
    # onnx need sample from the input data , here we take this sample from the data
    input_batch = next(iter(emotion_data.val_dataloader()))
    input_sample = {
    "input_ids" : input_batch["input_ids"][0].unsqueeze(0),
    "attention_mask" : input_batch["attention_mask"][0].unsqueeze(0)
    } 
    # let's export the model to onnx version
    torch.onnx.export(emotion_model , (input_sample["input_ids"] , input_sample["attention_mask"],) ,
    os.path.join(hydra.utils.get_original_cwd() , "models" , "model.onnx") ,
    export_params = True ,
    opset_version = 10 ,
    input_names = ["input_ids" , "attention_mask"] ,
    output_names = ["output"], 
    dynamic_axes ={
      "input_ids"      : {0 : "batch_size"} ,
      "attention_mask" : {0 : "batch_size"} , 
      "output"         : {0 : "batch_size"} ,
      }
    )

if __name__ == "__main__":
    from_torch_to_onnx()
