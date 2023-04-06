from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader


from monai.transforms import (
    AddChannel,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
)

from InferenceLib.data_utils.utils import get_image_files_single_scalar
from InferenceLib.data_utils.CustomDataset import GeomCnnDataset


class GeomCnnDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = -1,
                 val_frac: float = 0.2,
                 num_workers=4,
                 data_tuple=None,
                 file_paths=None):
        super(GeomCnnDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.FILE_PATHS = file_paths
        self.test_transform = Compose(
                [LoadImage(image_only=True), AddChannel(), ScaleIntensity(), EnsureType()]
            )
        self.data_tuple = data_tuple
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        print("Setting up data loaders ...")
        test_files, test_labels = get_image_files_single_scalar("TEST_DATA_DIR", self.FILE_PATHS)
        self.test_ds = GeomCnnDataset(test_files, test_labels, self.test_transform)
        print("Finished loading !!!")

    def test_dataloader(self):
        self.setup()
        return DataLoader(self.test_ds, self.batch_size)
