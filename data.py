import torch
import datasets
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer , BertTokenizer


class Data(pl.LightningDataModule):
    def __init__(self , cfg):
        super().__init__()
        self.tokenizer_name = cfg.dataset.TOKENIZER
        self.batch_size = cfg.dataset.BATCHSIZE
        self.max_length = cfg.dataset.MAXLENGTH
        self.tokenizer  = BertTokenizer.from_pretrained(self.tokenizer_name)

    def prepare_data(self):
        emotion_dataset = load_dataset("emotion")

        self.train_data = emotion_dataset["train"]
        self.valid_data   = emotion_dataset["validation"]
        self.test_data  = emotion_dataset["test"]

    def tokenize_data(self , text):
        return self.tokenizer(text["text"] , truncation = True , padding = "max_length" , max_length = self.max_length)

    def setup(self , stage = None):
        self.train_data = self.train_data.map(self.tokenize_data , batched = True)
        self.train_data.set_format(type = "torch" , columns = ["input_ids" , "attention_mask" , "label"]  , output_all_columns=True )

        self.valid_data = self.valid_data.map(self.tokenize_data , batched = True)
        self.valid_data.set_format(type = "torch" , columns = ["input_ids" , "attention_mask" , "label"]  , output_all_columns=True)

        self.test_data = self.test_data.map(self.tokenize_data , batched = True)
        self.test_data.set_format(type = "torch"  , columns = ["input_ids" , "attention_mask" , "label"]  , output_all_columns=True)

    def train_dataloader(self):
        return DataLoader(self.train_data , batch_size = self.batch_size , shuffle = True)
    def val_dataloader(self):
        return DataLoader(self.valid_data , batch_size = self.batch_size , shuffle = False)
    def test_dataloader(self):
        return DataLoader(self.test_data , batch_size = self.batch_size  , shuffle = False)
