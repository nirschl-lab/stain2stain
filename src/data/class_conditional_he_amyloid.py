import os
import random
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from PIL import Image, PngImagePlugin

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
from lightning import LightningDataModule
PngImagePlugin.MAX_TEXT_CHUNK = 10 * (1024**2)  # 10MB instead of default 1MB


class PairedAnyToAnyDataset(Dataset):
    """
    Any-to-any mapping dataset where each domain folder contains images with the SAME FILENAMES.
    For a given filename, returns (source_img from some domain, target_img from a random domain, target_label).

    Returns:
        source_img: Tensor (C,H,W)
        target_img: Tensor (C,H,W) with same filename, from target domain
        target_label: int (target domain class index)
    """

    def __init__(
        self,
        root_dir,
        class_folder_mapping,
        crop_size=256,
        transform=None,
        same_crop_for_pair=True,
        source_domain_mode="random",   # "random" or an int class_idx to fix source domain
        filename_mode="intersection",  # "intersection" (safe) or "union" (keeps all, checks presence)
        allowed_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        valid_filenames: Optional[List[str]] = None,  # If provided, only use these filenames
    ):
        self.root_dir = root_dir
        self.class_folder_mapping = dict(class_folder_mapping)
        self.crop_size = crop_size
        self.same_crop_for_pair = same_crop_for_pair
        self.source_domain_mode = source_domain_mode
        self.filename_mode = filename_mode
        self.allowed_exts = tuple(allowed_exts)

        self.num_classes = len(self.class_folder_mapping)
        self.class_indices = sorted(self.class_folder_mapping.keys())

        # Collect filenames per class
        self.class_to_dir = {
            c: os.path.join(root_dir, folder) for c, folder in self.class_folder_mapping.items()
        }

        self.class_to_filenames = {}
        for c, d in self.class_to_dir.items():
            if not os.path.isdir(d):
                raise ValueError(f"Folder not found: {d}")

            files = [
                f for f in os.listdir(d)
                if f.lower().endswith(self.allowed_exts)
            ]
            self.class_to_filenames[c] = set(files)
            print(f"Class {c} ({self.class_folder_mapping[c]}): {len(files)} files")

        # Decide which filenames are valid indices
        sets = list(self.class_to_filenames.values())
        if filename_mode == "intersection":
            common = set.intersection(*sets) if sets else set()
            all_filenames = sorted(list(common))
        elif filename_mode == "union":
            union = set.union(*sets) if sets else set()
            all_filenames = sorted(list(union))
        else:
            raise ValueError("filename_mode must be 'intersection' or 'union'")

        # Filter by valid_filenames if provided
        if valid_filenames is not None:
            self.filenames = sorted([f for f in all_filenames if f in valid_filenames])
        else:
            self.filenames = all_filenames

        if len(self.filenames) == 0:
            raise ValueError("No filenames found (check folders / extensions).")

        print(f"Using {len(self.filenames)} filenames ({filename_mode}).")

        # Post-crop transform
        if transform is None:
            self.post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.post_transform = transform

    def __len__(self):
        return len(self.filenames)

    def _load_rgb(self, class_idx, filename):
        path = os.path.join(self.class_to_dir[class_idx], filename)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # choose source domain
        if self.source_domain_mode == "random":
            source_label = random.choice(self.class_indices)
        elif isinstance(self.source_domain_mode, int):
            source_label = self.source_domain_mode
        else:
            raise ValueError("source_domain_mode must be 'random' or an int class index")

        # choose target domain (can be same as source)
        target_label = random.choice(self.class_indices)

        # If union mode, make sure chosen domains actually have this fname
        if self.filename_mode == "union":
            # resample until both exist (usually fast if mostly complete)
            tries = 0
            while (fname not in self.class_to_filenames[source_label]) or (fname not in self.class_to_filenames[target_label]):
                source_label = random.choice(self.class_indices) if self.source_domain_mode == "random" else source_label
                target_label = random.choice(self.class_indices)
                tries += 1
                if tries > 50:
                    raise RuntimeError(f"Could not find paired file '{fname}' across sampled domains. Consider using intersection mode.")

        src = self._load_rgb(source_label, fname)
        tgt = self._load_rgb(target_label, fname)

        # same crop coords for alignment
        if self.same_crop_for_pair:
            i, j, h, w = transforms.RandomCrop.get_params(src, output_size=(self.crop_size, self.crop_size))
            src = TF.crop(src, i, j, h, w)
            tgt = TF.crop(tgt, i, j, h, w)
        else:
            src = transforms.RandomCrop((self.crop_size, self.crop_size))(src)
            tgt = transforms.RandomCrop((self.crop_size, self.crop_size))(tgt)

        src = self.post_transform(src)
        tgt = self.post_transform(tgt)

        return src, tgt, target_label

class ClassConditionalAnyToAnyDataModule(LightningDataModule):
    """
    LightningDataModule for any-to-any stain transfer with class conditioning.
    Expects separate train/val/test directories with consistent structure.
    """

    def __init__(
        self,
        data_dir: str,
        class_folder_mapping: dict,
        crop_size: int = 256,
        same_crop_for_pair: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        source_domain_mode: str = "random",
        filename_mode: str = "intersection",
        allowed_exts: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        val_split: float = 0.2,
        split_seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.class_folder_mapping = class_folder_mapping
        self.crop_size = crop_size
        self.same_crop_for_pair = same_crop_for_pair
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.source_domain_mode = source_domain_mode
        self.filename_mode = filename_mode
        self.allowed_exts = allowed_exts
        self.val_split = val_split
        self.split_seed = split_seed
        
        # Will be populated in prepare_data
        self.train_filenames = None
        self.val_filenames = None
        self.split_file = Path(data_dir) / "train_val_split.json"


    def prepare_data(self) -> None:
        """Create train/val split if it doesn't exist.
        Lightning ensures that `self.prepare_data()` is called only within a single process on CPU.
        """
        # Check if split file already exists
        if self.split_file.exists():
            print(f"Split file already exists: {self.split_file}")
            return
        
        print(f"Creating train/val split with {self.val_split*100}% validation...")
        
        # Get all filenames from one of the domains to determine the full dataset
        first_class = list(self.class_folder_mapping.keys())[0]
        first_folder = self.class_folder_mapping[first_class]
        folder_path = os.path.join(self.data_dir, first_folder)
        
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        
        all_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(self.allowed_exts)
        ])
        
        if len(all_files) == 0:
            raise ValueError(f"No files found in {folder_path}")
        
        # Create reproducible split
        rng = random.Random(self.split_seed)
        rng.shuffle(all_files)
        
        n_val = int(len(all_files) * self.val_split)
        val_files = all_files[:n_val]
        train_files = all_files[n_val:]
        
        # Save split to file
        split_data = {
            "train": train_files,
            "val": val_files,
            "split_seed": self.split_seed,
            "val_split": self.val_split,
            "total_files": len(all_files),
            "train_files": len(train_files),
            "val_files": len(val_files)
        }
        
        with open(self.split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"Split created: {len(train_files)} train, {len(val_files)} val")
        print(f"Split saved to: {self.split_file}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Load split from file
        if not self.split_file.exists():
            raise RuntimeError(f"Split file not found: {self.split_file}. Make sure prepare_data() was called.")
        
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)
        
        self.train_filenames = split_data["train"]
        self.val_filenames = split_data["val"]
        
        print(f"Loaded split: {len(self.train_filenames)} train, {len(self.val_filenames)} val files")
        
        # Create datasets
        self.data_train = PairedAnyToAnyDataset(
            root_dir=self.data_dir,
            class_folder_mapping=self.class_folder_mapping,
            crop_size=self.crop_size,
            same_crop_for_pair=self.same_crop_for_pair,
            source_domain_mode=self.source_domain_mode,
            filename_mode=self.filename_mode,
            allowed_exts=self.allowed_exts,
            valid_filenames=self.train_filenames,
        )
        
        self.data_val = PairedAnyToAnyDataset(
            root_dir=self.data_dir,
            class_folder_mapping=self.class_folder_mapping,
            crop_size=self.crop_size,
            same_crop_for_pair=self.same_crop_for_pair,
            source_domain_mode=self.source_domain_mode,
            filename_mode=self.filename_mode,
            allowed_exts=self.allowed_exts,
            valid_filenames=self.val_filenames,
        )
        
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
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
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
    # Example usage
    data_module = ClassConditionalAnyToAnyDataModule(
        data_dir="/data1/shared/data/destain_restain/he_amyloid",
        class_folder_mapping={
            0: "train_he",
            1: "train_ihc",
            2: "train_gray"
        },
        crop_size=256,
        batch_size=16,
        num_workers=4,
        val_split=0.2,
        split_seed=123,
    )
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    for batch in train_loader:
        src, tgt, tgt_label = batch
        print(f"Train batch - src: {src.shape}, tgt: {tgt.shape}, tgt_label: {tgt_label}")
        break

    for batch in val_loader:
        src, tgt, tgt_label = batch
        print(f"Val batch - src: {src.shape}, tgt: {tgt.shape}, tgt_label: {tgt_label}")
        break
    