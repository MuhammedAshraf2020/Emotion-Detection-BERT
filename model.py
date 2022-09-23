import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
import torch.nn.functional as F

import torchmetrics
from sklearn.metrics import confusion_matrix

from transformers import BertModel
from sklearn.metrics import accuracy_score

class Model(pl.LightningModule):
    def __init__(self ,cfg):
        super(Model , self).__init__()
        self.cfg = cfg
        model_name=  self.cfg.model.MODELNAME 
        number_of_classes = self.cfg.model.CLASSESNUM
        self.base = BertModel.from_pretrained(model_name , return_dict = True)

        self.task = nn.Sequential(
          nn.Linear(self.base.config.hidden_size , number_of_classes))

        self.num_classes = number_of_classes

        self.train_accuracy_metric  = torchmetrics.Accuracy()
        self.val_accuracy_metric    = torchmetrics.Accuracy()
        self.test_accuracy_metric   = torchmetrics.Accuracy()

        self.f1_metric              = torchmetrics.F1Score(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(average="macro", num_classes=self.num_classes)
        self.recall_macro_metric    = torchmetrics.Recall(average="macro", num_classes=self.num_classes)
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric    = torchmetrics.Recall(average="micro")

    def forward(self , input_ids , attention_mask):
        outputs = self.base(input_ids = input_ids , attention_mask = attention_mask )
        logits  = self.task(outputs.pooler_output)
        return logits

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.cfg.model.LR ,  eps = self.cfg.model.EPS)

    def training_step(self, batch, batch_idx):
        logits = self(input_ids = batch["input_ids"]  , attention_mask = batch["attention_mask"] )
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(input_ids = batch["input_ids"]  , attention_mask = batch["attention_mask"] )


        labels = batch["label"]
        _, preds = torch.max(logits, dim=1)


        f1              = self.f1_metric(preds, labels)
        loss            = F.cross_entropy(logits, labels)
        valid_acc       = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro    = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro    = self.recall_micro_metric(preds, labels)

        self.log("val_loss"              , loss            , prog_bar=True, on_epoch=True)
        self.log("valid_acc"             , valid_acc       , prog_bar=True, on_epoch=True)
        self.log("valid_precision_macro" , precision_macro , prog_bar=True, on_epoch=True)
        self.log("valid_recall_macro"    , recall_macro    , prog_bar=True, on_epoch=True)
        self.log("valid_precision_micro" , precision_micro , prog_bar=True, on_epoch=True)
        self.log("valid_recall_micro"    , recall_micro    , prog_bar=True, on_epoch=True)
        self.log("valid_f1"              , f1              , prog_bar=True, on_epoch=True)


    def test_step(self , batch , batch_idx):
        logits = self.base(input_ids = batch["input_ids"]  , attention_mask = batch["attention_mask"] )
        loss = F.cross_entropy(logits , batch["label"])

        labels = batch["label"]
        _ , preds = torch.max(logits , dim = 1)


        f1              = self.f1_metric(preds, labels)
        loss            = F.cross_entropy(logits, labels)
        test_acc        = self.test_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro    = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro    = self.recall_micro_metric(preds, labels)

        self.log("test_loss"            , loss            , prog_bar=True, on_epoch=True)
        self.log("test_acc"             , test_acc        , prog_bar=True, on_epoch=True)
        self.log("test_precision_macro" , precision_macro , prog_bar=True, on_epoch=True)
        self.log("test_recall_macro"    , recall_macro    , prog_bar=True, on_epoch=True)
        self.log("test_precision_micro" , precision_micro , prog_bar=True, on_epoch=True)
        self.log("test_recall_micro"    , recall_micro    , prog_bar=True, on_epoch=True)
        self.log("test_f1"              , f1              , prog_bar=True, on_epoch=True)
