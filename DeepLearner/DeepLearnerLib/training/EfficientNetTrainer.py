import logging
import os.path
from argparse import ArgumentParser

import slicer

from DeepLearnerLib.Asynchrony import Asynchrony

try:
    import pytorch_lightning as pl
except ImportError:
    slicer.util.pip_install('pytorch_lightning==1.4.9')
    import pytorch_lightning as pl

try:
    import torch
except ImportError:
    slicer.util.pip_install('torch==1.9.0')
    import torch

try:
    import monai
except ImportError:
    slicer.util.pip_install('monai==0.7.0')

try:
    import pandas
except ImportError:
    slicer.util.pip_install('pandas==1.1.5')

try:
    import torchmetrics
except ImportError:
    slicer.util.pip_install('torchmetrics==0.6.0')

try:
    import sklearn
except ImportError:
    slicer.util.pip_install('scikit-learn==0.24.2')


import torch.nn
from monai.networks.nets import EfficientNetBN, DenseNet121, DenseNet, SEResNet50

from DeepLearnerLib.models.cnn_model import SimpleCNN
from DeepLearnerLib.pl_modules.classifier_modules import ImageClassifier
from DeepLearnerLib.data_utils.GeomCnnDataset import GeomCnnDataModule, GeomCnnDataModuleKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBarBase


def weight_reset(m):
    if isinstance(m, torch.nn.Module) and hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def setProgressBar(qtProgressBarObject, value):
    qtProgressBarObject.setValue(value)


class LitProgressBar(ProgressBarBase):
    def __init__(self, qtProgressBarObject):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.qtProgressBarObject = qtProgressBarObject

    def disable(self):
        self.enable = False

    def on_train_epoch_end(self, trainer, pl_module, **kwargs):
        super().on_train_epoch_end(trainer, pl_module)  # don't forget this :)
        percent = (pl_module.current_epoch + 1) * 100.0 / trainer.max_epochs
        Asynchrony.RunOnMainThread(lambda: setProgressBar(self.qtProgressBarObject, percent))


def cli_main(args):
    # pl.seed_everything(1234)

    # ------------
    # model
    # ------------
    if args["model"] == "eff_bn":
        backbone = EfficientNetBN(
            model_name="efficientnet-b0",
            in_channels=args["in_channels"],
            pretrained=True,
            num_classes=2
        )
    elif args["model"] == "densenet":
        backbone = DenseNet(
            spatial_dims=2,
            in_channels=args["in_channels"],
            out_channels=2
        )
    elif args["model"] == "resnet":
        backbone = SEResNet50(
            spatial_dims=2,
            in_channels=args["in_channels"],
            num_classes=2,
            pretrained=True
        )
    else:
        backbone = SimpleCNN()
    device = "cuda:0" if torch.cuda.is_available() and args["use_gpu"] else "cpu"
    model = ImageClassifier(backbone, learning_rate=args["learning_rate"],
                            criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, args["pos_weight"]])),
                            device=device,
                            metrics=["acc", "precision", "recall"])

    # -----------
    # Data
    # -----------
    if args["n_folds"] == 1:
        data_modules = [
            GeomCnnDataModule(
                batch_size=args["batch_size"],
                num_workers=args["data_workers"],
                file_paths=args["file_paths"]
            )
        ]

    else:
        data_module_generator = GeomCnnDataModuleKFold(
            batch_size=args["batch_size"],
            num_workers=args["data_workers"],
            n_splits=args["n_folds"],
            file_paths=args["file_paths"]
        )
        data_modules = data_module_generator.get_folds()

    for i in range(args["n_folds"]):
        # logger
        logger = TensorBoardLogger(
            save_dir=os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i)),
            name=args["exp_name"]
        )
        # early stopping
        es = EarlyStopping(
            monitor='validation/valid_loss',
            patience=30
        )
        progressBar = LitProgressBar(args["qtProgressBarObject"])
        checkpointer = ModelCheckpoint(
            monitor=args["monitor"],
            save_top_k=args["maxCp"], verbose=True, save_last=False,
            every_n_epochs=args["cp_n_epoch"],
            dirpath=os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i), "checkpoints")
        )
        # ------------
        # training
        # ------------
        trainer = pl.Trainer(max_epochs=args["max_epochs"],
                             gpus=0 if device == "cpu" else 1,
                             log_every_n_steps=5,
                             num_sanity_val_steps=1,
                             logger=logger,
                             callbacks=[progressBar, checkpointer, es])
        trainer.fit(model, datamodule=data_modules[i])
        saved_name = os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i), "model.pt")
        logging.info(f"Saving model: {saved_name}")
        torch.save(model.backbone, saved_name)
        model.apply(weight_reset)
        if args["qtProgressBarObject"] is not None:
            Asynchrony.RunOnMainThread(lambda: setProgressBar(args["qtProgressBarObject"], 0.0))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--in_channels', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--data_workers', type=int, default=4)
    parser.add_argument('--model', type=str, default="simple_cnn")
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--write_dir', type=str, default="default_write_dir")
    parser.add_argument('--exp_name', type=str, default="default_name")
    parser.add_argument('--cp_n_epoch', type=int, default=1)
    parser.add_argument('--maxCp', type=int, default=2)
    parser.add_argument('--monitor', type=str, default="validation/valid_loss")
    parser.add_argument('--pos_weight', type=float, default=1.0)

    args = vars(parser.parse_args())
    args["qtProgressBarObject"] = None
    cli_main(args)