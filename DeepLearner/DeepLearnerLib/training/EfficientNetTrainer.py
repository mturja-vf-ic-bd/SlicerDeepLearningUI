import os.path
import sys
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
from monai.networks.nets import EfficientNetBN, DenseNet121, DenseNet

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
            pretrained=False,
            num_classes=2
        )
    elif args["model"] == "densenet":
        backbone = DenseNet(
            spatial_dims=2,
            in_channels=args["in_channels"],
            out_channels=2
        )
    else:
        backbone = SimpleCNN()
    device = "cuda:0" if torch.cuda.is_available() and args["use_gpu"] else "cpu"
    print(f"Using device: {device}")
    model = ImageClassifier(backbone, learning_rate=args["learning_rate"],
                            criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 5.0])),
                            device=device,
                            metrics=["acc", "precision", "recall"])

    # -----------
    # Data
    # -----------
    if args["n_folds"] == 1:
        data_modules = [GeomCnnDataModule(batch_size=args["batch_size"], num_workers=args["data_workers"], file_paths=args["file_paths"])]
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
            dirpath=os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i), "checkpoints"))
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
        model.apply(weight_reset)
        Asynchrony.RunOnMainThread(lambda: setProgressBar(args["qtProgressBarObject"], 0.0))
