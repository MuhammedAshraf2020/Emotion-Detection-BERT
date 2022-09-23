
import hydra , os 
from pytorch_lightning.core.saving import CHECKPOINT_PAST_HPARAMS_KEYS
import wandb
import torch
import pandas as pd
from model import Model
from data import Data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import early_stopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import WandbLogger


class VisualizeWrongSample(pl.Callback):
  def __init__(self , dataloader):
    super().__init__()
    self.dataloader = dataloader

  def on_validation_end(self , trainer , recModel):

    sample = next(iter(self.dataloader.val_dataloader()))
    sentences = sample["text"]

    output = recModel(sample["input_ids"].to("cuda"), sample["attention_mask"].to("cuda"))
    logits = torch.argmax(output , axis = 1)
    labels = sample["label"]

    df   = pd.DataFrame({"Text" : sentences , "Labels" : labels  , "Predicted" : logits.detach().cpu() })
    w_df = df[df["Labels"] != df["Predicted"]]
    trainer.logger.experiment.log(
        {
            "examples" : wandb.Table(dataframe =  w_df , allow_mixed_types = True) ,
            "global_step" : trainer.global_step
        }
    )

@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
  model = Model(cfg)
  data  = Data(cfg)
  print(os.path.join(hydra.utils.get_original_cwd() , cfg.training.CHECKPOINT_DIR_PATH))
  checkpoint_callback = ModelCheckpoint(
    dirpath = os.path.join(hydra.utils.get_original_cwd() , cfg.training.CHECKPOINT_DIR_PATH) ,
    filename = cfg.training.CHECKPOINT_FILENAME  , 
    monitor = cfg.training.CHECKPOINT_MONITOR , 
    mode    = cfg.training.CHECKPOINT_MODE
  )

  early_stopping = EarlyStopping(
    monitor = cfg.training.EARLYSTOPPING_MONITOR  , 
    patience = cfg.training.PATIENCE , verbose = True , mode = cfg.training.EARLYSTOPPING_MODE
  )

  wandb_logger = WandbLogger(project = "EmotionRecognition" , entity = "muhammed266")
  GPUS = torch.cuda.device_count()
  trainer = pl.Trainer(max_epochs = cfg.training.MAXEPOCHS ,logger=wandb_logger, gpus = GPUS ,
      callbacks=[checkpoint_callback, VisualizeWrongSample(data), early_stopping],
      log_every_n_steps= cfg.training.LOG_EVERY_N_STEPS ,
      deterministic = cfg.training.DETERMINISTIC)
  trainer.fit(model, data)
  wandb.finish()

if __name__ == "__main__":
  main()
