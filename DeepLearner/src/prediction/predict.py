import pytorch_lightning as pl
import torch
import torchmetrics

from src.data_utils.GeomCnnDataset import GeomCnnDataModule
from src.data_utils.utils import get_test_dataloader
from src.pl_modules.classifier_modules import ImageClassifier


def predict(pred_dataloader, model_ckpt):
    metric_acc = torchmetrics.AUROC()
    model = ImageClassifier.load_from_checkpoint(model_ckpt)
    for x, y in pred_dataloader:
        pred = model(x)
        metric_acc(pred[:,  1], y)
        print("pred:", pred)
        print("true:", y)
    output = metric_acc.compute()
    return output


if __name__ == '__main__':
    test_dataloader = get_test_dataloader()
    model_ckpt = "lightning_logs/version_40/checkpoints/epoch=999-step=999.ckpt"
    pred = predict(test_dataloader, model_ckpt)
    print(pred)

