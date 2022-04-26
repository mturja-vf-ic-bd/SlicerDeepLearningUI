from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics
from DeepLearnerLib.Asynchrony import Asynchrony


class ImageClassifier(pl.LightningModule):
    def __init__(self,
                 backbone,
                 criterion=nn.CrossEntropyLoss(),
                 learning_rate=1e-3,
                 metrics=["acc"],
                 device="cpu"):
        super(ImageClassifier, self).__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.criterion = criterion
        self.train_metrics = {}
        self.val_metrics = {}
        self.metric_names = metrics
        for m in metrics:
            if m == "acc":
                train_metric = torchmetrics.Accuracy().to(device)
                val_metric = torchmetrics.Accuracy().to(device)
            elif m == "auc":
                train_metric = torchmetrics.AUROC(pos_label=1).to(device)
                val_metric = torchmetrics.AUROC(pos_label=1).to(device)
            elif m == "precision":
                train_metric = torchmetrics.Precision().to(device)
                val_metric = torchmetrics.Precision().to(device)
            elif m == "recall":
                train_metric = torchmetrics.Recall().to(device)
                val_metric = torchmetrics.Recall().to(device)
            self.train_metrics[m] = train_metric
            self.val_metrics[m] = val_metric

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def common_step(self, batch, batch_idx, mode="train"):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        y_hat = nn.Softmax(dim=-1)(y_hat)[:, 1]
        if mode == "train":
            for m in self.metric_names:
                self.train_metrics[m](y_hat, y)
        elif mode == "valid":
            for m in self.metric_names:
                self.val_metrics[m](y_hat, y)
        return loss

    def training_epoch_end(self, outputs):
        # update and log
        for m in self.metric_names:
            self.log(f"train/{m}", self.train_metrics[m].compute())
            self.train_metrics[m].reset()

    def validation_epoch_end(self, outputs):
        # update and log
        for m in self.metric_names:
            self.log(f"validation/{m}", self.val_metrics[m].compute())
            self.val_metrics[m].reset()

    def log_scalars(self, scalar_name, scalar_value):
        self.log(scalar_name, scalar_value, on_step=True)

    def training_step(self, batch, batch_idx):
        train_loss = self.common_step(batch, batch_idx, mode="train")
        self.log_scalars('train/train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss = self.common_step(batch, batch_idx, mode="valid")
        self.log_scalars('validation/valid_loss', valid_loss)
        return valid_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.common_step(batch, batch_idx, mode="test")
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_name', type=str, default="efficientnet-b0")
        parser.add_argument('--pretrained', action='store_true', default=False)
        return parser
