import os.path
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch.nn
from monai.networks.nets import EfficientNetBN, DenseNet121, DenseNet

from src.models.cnn_model import SimpleCNN
from src.pl_modules.classifier_modules import ImageClassifier
from src.data_utils.GeomCnnDataset import GeomCnnDataModule, GeomCnnDataModuleKFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ImageClassifier(backbone, learning_rate=args["learning_rate"],
                            criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 5.0])),
                            device=device,
                            metrics=["acc", "precision", "recall"])

    # -----------
    # Data
    # -----------
    # data_module = GeomCnnDataModule(batch_size=args.batch_size, num_workers=args.data_workers)
    data_module_generator = GeomCnnDataModuleKFold(
        batch_size=args["batch_size"],
        num_workers=args["data_workers"],
        n_splits=args["n_folds"]
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
        checkpointer = ModelCheckpoint(
            monitor=args["monitor"],
            save_top_k=args["maxCp"], verbose=True, save_last=False,
            every_n_epochs=args["cp_n_epoch"],
            dirpath=os.path.join(args["write_dir"], "logs", args["model"], "fold_" + str(i), "checkpoints"))
        # ------------
        # training
        # ------------
        trainer = pl.Trainer(max_epochs=args["max_epochs"],
                             gpus=args["gpus"],
                             log_every_n_steps=5,
                             num_sanity_val_steps=0,
                             logger=logger,
                             callbacks=[es, checkpointer])
        trainer.fit(model, datamodule=data_modules[i])

#
# if __name__ == '__main__':
#     cli_main()
