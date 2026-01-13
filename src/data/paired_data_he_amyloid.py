import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Optional, List
import pdb
import cv2
import numpy as np
from lightning import LightningDataModule

class PairedHEIHCDataset(Dataset):
    """
    Dataset class for paired HE and IHC images.
    Assumes that images are stored in two separate directories with matching filenames.
    """
    
    def __init__(self, data_dir, csv_file_name, folder, image_size=512, direction = "HE_to_IHC",):
        """
        Args:
            data_dir (str): Path to the data directory which contains csv file, train, test and val folder.
            csv_file_name (str): Name of the CSV file containing metadata.
                image_id: Unique identifier for each image pair.
                he_filepath: image_id + '_he.png'
                ihc_filepath: image_id + '_ihc.png'
            folder (str): One of 'train', 'val', or 'test' to specify the dataset split.
        """
        self.he_dir = os.path.join(data_dir, folder)
        self.ihc_dir = os.path.join(data_dir, folder)
        self.image_size = image_size
        self.direction = direction

        # Load metadata from CSV
        csv_path = os.path.join(data_dir, csv_file_name)
        self.metadata = pd.read_csv(csv_path)

        # Filter metadata for the specified folder
        self.metadata = self.metadata[self.metadata['split'] == folder].reset_index(drop=True)

        print(f"Loading paired HE-IHC dataset from:")
        print(f"  HE directory: {self.he_dir}")
        print(f"  IHC directory: {self.ihc_dir}")
        print(f"  Number of paired images: {len(self.metadata)}")

        # Default transform: resize and normalize
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Returns:
            he_img: HE image tensor (C, H, W)
            ihc_img: IHC image tensor (C, H, W)
            filename: Original filename for reference
        """
        row = self.metadata.iloc[idx]
        he_filename = row['he_filepath']
        ihc_filename = row['ihc_filepath']

        # Load HE image
        he_path = os.path.join(self.he_dir, he_filename)
        assert os.path.exists(he_path), f"HE image not found: {he_path}"
        he_img_cv = cv2.imread(he_path)
        he_img = Image.fromarray(cv2.cvtColor(he_img_cv, cv2.COLOR_BGR2RGB))

        # Load IHC image
        ihc_path = os.path.join(self.ihc_dir, ihc_filename)
        assert os.path.exists(ihc_path), f"IHC image not found: {ihc_path}"
        ihc_img_cv = cv2.imread(ihc_path)
        ihc_img = Image.fromarray(cv2.cvtColor(ihc_img_cv, cv2.COLOR_BGR2RGB))
        # Apply transformations
        he_img = self.transform(he_img)
        ihc_img = self.transform(ihc_img)

        if self.direction == "HE_to_IHC":
            return he_img, ihc_img
        else:
            return ihc_img, he_img # Reverse direction

class PairedHEIHCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        csv_file_name: str = "dataset_nirschl_et_al_2026_metadata.csv",
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 512,
        direction: str = "HE_to_IHC",
        pin_memory: bool = True
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.csv_file_name = csv_file_name
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.direction = direction
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        self.data_train = PairedHEIHCDataset(
            data_dir=self.data_dir,
            csv_file_name=self.csv_file_name,
            folder="train",
            image_size=self.image_size,
            direction=self.direction,
        )
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        self.data_val = PairedHEIHCDataset(
            data_dir=self.data_dir,
            csv_file_name=self.csv_file_name,
            folder="val",
            image_size=self.image_size,
            direction=self.direction,
        )
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        self.data_test = PairedHEIHCDataset(
            data_dir=self.data_dir,
            csv_file_name=self.csv_file_name,
            folder="test",
            image_size=self.image_size,
            direction=self.direction,
        )
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    data_dir = "/data1/shared/data/destain_restain/he_amyloid/positive_crops/flow_matching/"
    dataset = PairedHEIHCDataset(data_dir=data_dir, csv_file_name="dataset_nirschl_et_al_2026_metadata.csv", folder="train", image_size=512)
    print(f"Dataset length: {len(dataset)}")
    he_img, ihc_img = dataset[0]
    print(f"HE image shape: {he_img.shape}, IHC image shape: {ihc_img.shape}")

    data_module = PairedHEIHCDataModule(data_dir=data_dir, csv_file_name="dataset_nirschl_et_al_2026_metadata.csv", batch_size=4, image_size=512)
    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        he_imgs, ihc_imgs = batch
        print(f"Batch HE images shape: {he_imgs.shape}, Batch IHC images shape: {ihc_imgs.shape}")
        break
    val_loader = data_module.val_dataloader()
    for batch in val_loader:
        he_imgs, ihc_imgs = batch
        print(f"Batch HE images shape: {he_imgs.shape}, Batch IHC images shape: {ihc_imgs.shape}")
        break
    test_loader = data_module.test_dataloader()
    for batch in test_loader:
        he_imgs, ihc_imgs = batch
        print(f"Batch HE images shape: {he_imgs.shape}, Batch IHC images shape: {ihc_imgs.shape}")
        break