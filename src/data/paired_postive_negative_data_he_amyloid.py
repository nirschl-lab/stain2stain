import os
import pandas as pd
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from typing import Any, Dict, Optional, List
import pdb
import cv2
import numpy as np
from lightning import LightningDataModule

PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024**2)  # 10 MB (adjust as needed)

class PairedNegativeHEIHCDataModule(Dataset):
    """
    Dataset for paired HE (Hematoxylin and Eosin) and IHC (Immunohistochemistry) images.
    
    Args:
        he_dir: Path to directory containing HE images
        ihc_dir: Path to directory containing IHC images
        crop_size: Size of random crop to extract (default: 256)
        transform: Optional transform to apply to both images
    """
    
    def __init__(self, he_dir, ihc_dir, crop_size=512, direction = "HE_to_IHC"):
        self.he_dir = he_dir
        self.ihc_dir = ihc_dir
        self.crop_size = crop_size
        self.direction = direction

        print(f"Loading paired HE-IHC dataset from:")
        print(f"  HE directory: {he_dir}")
        print(f"  IHC directory: {ihc_dir}")
        
        # Get all image files from HE directory
        self.he_files = sorted([f for f in os.listdir(he_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        # Get all image files from IHC directory
        self.ihc_files = sorted([f for f in os.listdir(ihc_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        # Ensure paired images have matching filenames
        self.he_files_set = set(self.he_files)
        self.ihc_files_set = set(self.ihc_files)
        
        # Find intersection of filenames
        common_files = self.he_files_set.intersection(self.ihc_files_set)
        self.image_files = sorted(list(common_files))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No matching paired images found between {he_dir} and {ihc_dir}")
        
        print(f"Found {len(self.image_files)} paired HE-IHC images")
        
        
        self.transform = transforms.Compose([
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            he_img: HE image tensor (C, H, W)
            ihc_img: IHC image tensor (C, H, W)
            filename: Original filename for reference
        """
        filename = self.image_files[idx]
        
        # Load HE image
        he_path = os.path.join(self.he_dir, filename)
        he_img = Image.open(he_path).convert('RGB')
        
        # Load IHC image
        ihc_path = os.path.join(self.ihc_dir, filename)
        ihc_img = Image.open(ihc_path).convert('RGB')
        
        # Apply same random crop to both images by setting the same seed
        seed = torch.randint(0, 2**32, (1,)).item()
        
        torch.manual_seed(seed)
        he_img = self.transform(he_img)
        
        torch.manual_seed(seed)
        ihc_img = self.transform(ihc_img)

        if self.direction == "HE_to_IHC":
            return he_img, ihc_img
        else:
            return ihc_img, he_img

class PairedPostiveHEIHCDataset(Dataset):
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
        pin_memory: bool = True,
        negative_data_dir: Optional[str] = None,
        negative_he_folder: str = "train_he",
        negative_ihc_folder: str = "train_ihc",
        crop_size: int = 512,
        use_negative_data: bool = False,
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
        self.negative_data_dir = negative_data_dir
        self.negative_he_folder = negative_he_folder
        self.negative_ihc_folder = negative_ihc_folder
        self.crop_size = crop_size
        self.use_negative_data = use_negative_data

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
        # Load positive dataset
        positive_dataset = PairedPostiveHEIHCDataset(
            data_dir=self.data_dir,
            csv_file_name=self.csv_file_name,
            folder="train",
            image_size=self.image_size,
            direction=self.direction,
        )
        
        if self.use_negative_data and self.negative_data_dir is not None:
            # Load negative dataset
            negative_he_dir = os.path.join(self.negative_data_dir, self.negative_he_folder)
            negative_ihc_dir = os.path.join(self.negative_data_dir, self.negative_ihc_folder)
            negative_dataset = PairedNegativeHEIHCDataModule(
                he_dir=negative_he_dir,
                ihc_dir=negative_ihc_dir,
                crop_size=self.crop_size,
                direction=self.direction,
            )
            
            # Combine datasets
            self.data_train = ConcatDataset([positive_dataset, negative_dataset])
            
            # Create balanced sampling weights
            num_positive = len(positive_dataset)
            num_negative = len(negative_dataset)
            print(f"Training dataset composition: Positive={num_positive}, Negative={num_negative}")
            
            # Assign weights: lower weight for positive (more samples), higher weight for negative (fewer samples)
            # This will oversample the negative dataset to balance with positive
            weight_positive = 1.0 / num_positive
            weight_negative = 1.0 / num_negative
            
            # Create sample weights for each sample in the combined dataset
            sample_weights = [weight_positive] * num_positive + [weight_negative] * num_negative
            
            # Create weighted sampler for balanced sampling
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.data_train),
                replacement=True  # Allow oversampling
            )
            
            print(f"Using balanced sampling with oversampling for negative samples")
            
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                sampler=sampler,  # Use sampler instead of shuffle
            )
        else:
            # Only positive data
            self.data_train = positive_dataset
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
        self.data_val = PairedPostiveHEIHCDataset(
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
        self.data_test = PairedPostiveHEIHCDataset(
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
    positve_data_dir = "/data1/shared/data/destain_restain/he_amyloid/positive_crops/flow_matching/"
    negative_data_dir = "/data1/shared/data/destain_restain/he_amyloid/"
    he_folder = 'train_he'
    ihc_folder = 'train_ihc'
    negative_he_dir = os.path.join(negative_data_dir, he_folder)
    negative_ihc_dir = os.path.join(negative_data_dir, ihc_folder)

    # check positve dataset
    dataset_positive = PairedPostiveHEIHCDataset(data_dir=positve_data_dir, csv_file_name="dataset_nirschl_et_al_2026_metadata.csv", folder="train", image_size=512)
    print(f"Postive Dataset length: {len(dataset_positive)}")
    he_img, ihc_img = dataset_positive[0]
    print(f"HE image shape: {he_img.shape}, IHC image shape: {ihc_img.shape}")

    #check negative dataset
    dataset_negative = PairedNegativeHEIHCDataModule(negative_he_dir, negative_ihc_dir, crop_size=512, direction = "HE_to_IHC")
    print(f"Negative Dataset length: {len(dataset_negative)}")
    he_img, ihc_img = dataset_negative[0]
    print(f"HE image shape: {he_img.shape}, IHC image shape: {ihc_img.shape}")
    
    # Test combined dataset with balanced sampling using the updated PairedHEIHCDataModule
    print("\n" + "="*50)
    print("Testing PairedHEIHCDataModule with Balanced Sampling")
    print("="*50)
    
    data_module = PairedHEIHCDataModule(
        data_dir=positve_data_dir,
        csv_file_name="dataset_nirschl_et_al_2026_metadata.csv",
        batch_size=8,
        num_workers=0,
        image_size=512,
        negative_data_dir=negative_data_dir,
        negative_he_folder=he_folder,
        negative_ihc_folder=ihc_folder,
        crop_size=512,
        use_negative_data=True
    )
    
    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    
    # Sample batches to verify balanced sampling
    print("\nTesting balanced sampling over 50 batches:")
    num_test_batches = 2
    total_positive_samples = len(dataset_positive)
    total_negative_samples = len(dataset_negative)
    
    # Track sampling statistics
    positive_sample_counts = {}
    negative_sample_counts = {}
    batch_sizes = []
    
    for i, batch in enumerate(train_loader):
        if i >= num_test_batches:
            break
        he_imgs, ihc_imgs = batch
        batch_sizes.append(len(he_imgs))
        if i < 5:  # Print first 5 batches
            print(f"Batch {i+1}: HE shape={he_imgs.shape}, IHC shape={ihc_imgs.shape}")
    
    print(f"\nProcessed {num_test_batches} batches with average batch size: {np.mean(batch_sizes):.2f}")
    print(f"Total samples drawn: {sum(batch_sizes)}")
    print(f"\nDataset composition:")
    print(f"  Positive samples: {total_positive_samples}")
    print(f"  Negative samples: {total_negative_samples}")
    print(f"  Imbalance ratio: {total_positive_samples/total_negative_samples:.2f}x")
    print(f"\nWith balanced sampling, negative samples should be drawn ~{total_positive_samples/total_negative_samples:.1f}x more frequently")
    
    # Calculate expected sampling probabilities
    print(f"\nExpected sampling probabilities per epoch:")
    print(f"  Each positive sample: ~{100.0/total_positive_samples:.3f}%")
    print(f"  Each negative sample: ~{100.0/total_negative_samples:.3f}% (oversampled by ~{total_positive_samples/total_negative_samples:.1f}x)")
    
    print("\n" + "="*50)
    print("Testing validation and test dataloaders (positive samples only):")
    print("="*50)
    
    val_loader = data_module.val_dataloader()
    val_batch = next(iter(val_loader))
    he_imgs, ihc_imgs = val_batch
    print(f"Val batch: HE shape={he_imgs.shape}, IHC shape={ihc_imgs.shape}")
    print(f"Val dataset size: {len(data_module.data_val)}")
    
    test_loader = data_module.test_dataloader()
    test_batch = next(iter(test_loader))
    he_imgs, ihc_imgs = test_batch
    print(f"Test batch: HE shape={he_imgs.shape}, IHC shape={ihc_imgs.shape}")
    print(f"Test dataset size: {len(data_module.data_test)}")
    
    print("\n" + "="*50)
    print("Summary: Training uses balanced sampling with positive and negative data.")
    print("Val and test use only positive data (as specified).")
    print("="*50)