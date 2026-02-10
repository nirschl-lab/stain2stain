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
import random
from lightning import LightningDataModule
import torchvision.transforms.functional as TF


# class PairedHEIHCDataset(Dataset):
#     """
#     Dataset class for paired HE and IHC images.
#     Assumes that images are stored in two separate directories with matching filenames.
#     """
    
#     def __init__(self, data_dir, csv_file_name, source_column, target_column, folder, image_size=512, direction = "HE_to_IHC",):
#         """
#         Args:
#             data_dir (str): Path to the data directory which contains csv file, train, test and val folder.
#             csv_file_name (str): Name of the CSV file containing metadata.
#                 image_id: Unique identifier for each image pair.
#                 he_filepath: image_id + '_he.png'
#                 ihc_filepath: image_id + '_ihc.png'
#             folder (str): One of 'train', 'val', or 'test' to specify the dataset split.
#         """
#         self.he_dir = os.path.join(data_dir, folder)
#         self.ihc_dir = os.path.join(data_dir, folder)
#         self.image_size = image_size
#         self.direction = direction
#         self.source_column = source_column
#         self.target_column = target_column

#         # Load metadata from CSV
#         csv_path = os.path.join(data_dir, csv_file_name)
#         assert os.path.exists(csv_path),'csv not exists'
#         # print('csv path ------', csv_path)
#         self.metadata = pd.read_csv(csv_path)

#         # Filter metadata for the specified folder
#         self.metadata = self.metadata[self.metadata['split'] == folder].reset_index(drop=True)

#         print(f"Loading paired HE-IHC dataset from:")
#         print(f"  HE directory: {self.he_dir}")
#         print(f"  IHC directory: {self.ihc_dir}")
#         print(f"  Number of paired images: {len(self.metadata)}")

#         # Default transform: resize and normalize
#         self.transform = transforms.Compose([
#             transforms.Resize((self.image_size, self.image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ])
#     def __len__(self):
#         return len(self.metadata)
    
#     def __getitem__(self, idx):
#         """
#         Returns:
#             he_img: HE image tensor (C, H, W)
#             ihc_img: IHC image tensor (C, H, W)
#             filename: Original filename for reference
#         """
#         row = self.metadata.iloc[idx]
#         he_filename = row[self.source_column]
#         ihc_filename = row[self.target_column]

#         # Load HE image
#         he_path = os.path.join(self.he_dir, he_filename)
#         assert os.path.exists(he_path), f"HE image not found: {he_path}"
#         he_img_cv = cv2.imread(he_path)
#         he_img = Image.fromarray(cv2.cvtColor(he_img_cv, cv2.COLOR_BGR2RGB))

#         # Load IHC image
#         ihc_path = os.path.join(self.ihc_dir, ihc_filename)
#         assert os.path.exists(ihc_path), f"IHC image not found: {ihc_path}"
#         ihc_img_cv = cv2.imread(ihc_path)
#         ihc_img = Image.fromarray(cv2.cvtColor(ihc_img_cv, cv2.COLOR_BGR2RGB))
#         # Apply transformations
#         he_img = self.transform(he_img)
#         ihc_img = self.transform(ihc_img)

#         if self.direction == "HE_to_IHC":
#             return he_img, ihc_img
#         else:
#             return ihc_img, he_img # Reverse direction
        
class PairedDataset(Dataset):
    """
    Dataset class for paired source and target images.
    Assumes that images are stored in two separate directories with matching filenames.
    """
    
    def __init__(self, 
                 data_dir, 
                 csv_file_name, 
                 source_column,  
                 target_column, 
                 folder, 
                 image_size=512, 
                 direction = "S2T", 
                 use_augmentation=False,
                 return_filename=False):
        """
        Args:
            data_dir (str): Path to the data directory which contains csv file, train, test and val folder.
            csv_file_name (str): Name of the CSV file containing metadata.
                image_id: Unique identifier for each image pair.
                he_filepath: image_id + '_he.png'
                lhe_filepath: image_id + '_lhe.png'
            folder (str): One of 'train', 'val', or 'test' to specify the dataset split.
        """
        self.source_dir = os.path.join(data_dir, folder)
        self.target_dir = os.path.join(data_dir, folder)
        self.image_size = image_size
        self.direction = direction
        self.source_column = source_column
        self.target_column = target_column
        self.return_filename = return_filename

        # Load metadata from CSV
        csv_path = os.path.join(data_dir, csv_file_name)
        assert os.path.exists(csv_path),'csv not exists'
        # print('csv path ------', csv_path)
        self.metadata = pd.read_csv(csv_path)

        # Filter metadata for the specified folder
        self.metadata = self.metadata[self.metadata['split'] == folder].reset_index(drop=True)

        print(f"Loading paired dataset from:")
        print(f"  Source directory: {self.source_dir}")
        print(f"  Target directory: {self.target_dir}")
        print(f"  Number of paired images: {len(self.metadata)}")

        # Store augmentation flag
        self.use_augmentation = use_augmentation
        
        # Normalization transform
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Returns:
            source_img: Source image tensor (C, H, W)
            target_img: Target image tensor (C, H, W)
        """
        row = self.metadata.iloc[idx]
        source_filename = row[self.source_column]
        target_filename = row[self.target_column]

        # Load source image
        source_path = os.path.join(self.source_dir, source_filename)
        assert os.path.exists(source_path), f"Source image not found: {source_path}"
        source_img_cv = cv2.imread(source_path)
        source_img = Image.fromarray(cv2.cvtColor(source_img_cv, cv2.COLOR_BGR2RGB))

        # Load target image
        target_path = os.path.join(self.target_dir, target_filename)
        assert os.path.exists(target_path), f"Target image not found: {target_path}"
        target_img_cv = cv2.imread(target_path)
        target_img = Image.fromarray(cv2.cvtColor(target_img_cv, cv2.COLOR_BGR2RGB))

        # Apply transformations
        if self.use_augmentation:
            # Get random crop parameters
            i, j, h, w = transforms.RandomCrop.get_params(
                source_img, output_size=(self.image_size, self.image_size)
            )
            
            # Apply the same crop to all images
            source_img = TF.crop(source_img, i, j, h, w)
            target_img = TF.crop(target_img, i, j, h, w)
            
            # Random horizontal flip
            if random.random() > 0.5:
                source_img = TF.hflip(source_img)
                target_img = TF.hflip(target_img)
                
            
            # Random vertical flip
            if random.random() > 0.5:
                source_img = TF.vflip(source_img)
                target_img = TF.vflip(target_img)

            # Convert to tensor
            source_img = TF.to_tensor(source_img)
            target_img = TF.to_tensor(target_img)
            
            # Normalize RGB images only
            source_img = self.normalize(source_img)
            target_img = self.normalize(target_img)
        else:
            # Resize without augmentation
            source_img = TF.resize(source_img, (self.image_size, self.image_size))
            target_img = TF.resize(target_img, (self.image_size, self.image_size))
            
            # Convert to tensor
            source_img = TF.to_tensor(source_img)
            target_img = TF.to_tensor(target_img)
            
            # Normalize RGB images only
            source_img = self.normalize(source_img)
            target_img = self.normalize(target_img)
        

        if self.direction == "S2T":
            if self.return_filename:
                return source_img, target_img, source_filename, target_filename
            else:
                return source_img, target_img
        else:
            if self.return_filename:
                return target_img, source_img, target_filename, source_filename
            else:
                return target_img, source_img

class PairedDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        csv_file_name: str = "dataset_nirschl_et_al_2026_metadata.csv",
        source_column: str = 'he_filepath',
        target_column: str = 'ihc_filepath',
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 512,
        direction: str = "S2T", 
        pin_memory: bool = True,
        use_augmentation: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.csv_file_name = csv_file_name
        self.source_column = source_column
        self.target_column = target_column
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.direction = direction
        self.pin_memory = pin_memory
        self.use_augmentation = use_augmentation

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
        self.data_train = PairedDataset(
            data_dir=self.data_dir,
            csv_file_name=self.csv_file_name,
            source_column=self.source_column,
            target_column=self.target_column,
            folder="train",
            image_size=self.image_size,
            direction=self.direction,
            use_augmentation=self.use_augmentation,
        )
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        self.data_val = PairedDataset(
            data_dir=self.data_dir,
            csv_file_name=self.csv_file_name,
            source_column=self.source_column,
            target_column=self.target_column,
            folder="val",
            image_size=self.image_size,
            direction=self.direction,
            use_augmentation=self.use_augmentation,
        )
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        self.data_test = PairedDataset(
            data_dir=self.data_dir,
            csv_file_name=self.csv_file_name,
            source_column=self.source_column,
            target_column=self.target_column,
            folder="test",
            image_size=self.image_size,
            direction=self.direction,
            use_augmentation=self.use_augmentation,
        )
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
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
    image_size = 256
    use_augmentation = True
    source_column = 'he_filepath'
    target_column = 'ihc_filepath'
    dataset = PairedDataset(data_dir=data_dir, csv_file_name="dataset_nirschl_et_al_2026_metadata.csv", folder="train", source_column=source_column, target_column=target_column, image_size=256, use_augmentation=use_augmentation)
    print(f"Dataset length: {len(dataset)}")
    he_img, ihc_img = dataset[0]
    print(f"HE image shape: {he_img.shape}, IHC image shape: {ihc_img.shape}")

    data_module = PairedHEIHCDataModule(data_dir=data_dir, csv_file_name="dataset_nirschl_et_al_2026_metadata.csv", batch_size=4, image_size=256, direction="HE_to_IHC", use_augmentation=use_augmentation, source_column=source_column, target_column=target_column)
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