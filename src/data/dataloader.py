"""
## Information

- File Name: dataloader.py
- Author: Selene
- Date of Creation: 2023.03.20
- Date of Last Modification: 2023.05.17 (TODO: Update this)
- Python Version: 3.9.13
- License: GNU GPL v3.0
"""

import glob
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class GeoLifeDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, data: List[Tensor]):
        self.data = data  # [X; (traj_length, 4)] (in order)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GeoIndDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, real: List[Tensor], synthetic: List[Tensor]):
        self.real = real  # [X; (traj_length, 4)] (in order)
        self.synthetic = synthetic  # [X; (traj_length, 4)] (in order)

    def __len__(self):
        return len(self.real)

    def __getitem__(self, idx):
        return self.real[idx], self.synthetic[idx]


@dataclass
class DataModuleConfig:
    """_summary_"""

    data_dir: str
    ind_dir: str
    batch_size: int
    length_range: List[int]
    ratio: List[float]
    num_workers: int


class GeoLifeDataModule(pl.LightningDataModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """

    def __init__(self, dm_config: DataModuleConfig):
        super().__init__()

        self.dm_config = dm_config
        self.dataset: GeoLifeDataset
        self.train_dataset: Subset
        self.val_dataset: Subset

    def prepare_data(self):
        min_length = self.dm_config.length_range[0]
        max_length = self.dm_config.length_range[1]

        trajectory_files = glob.glob(
            os.path.join(self.dm_config.data_dir, "**/*.csv"), recursive=True
        )

        trajectories: List[torch.Tensor] = []
        for file in trajectory_files:
            traj_csv = np.loadtxt(file, delimiter=",")
            # shape: (traj_length, 4)
            trajectory = torch.tensor(traj_csv, dtype=torch.float)
            if torch.isnan(trajectory).any():
                continue
            traj_length = trajectory.shape[0]
            if traj_length > max_length:
                chunked = torch.chunk(
                    trajectory, (traj_length // max_length) + 1, dim=0
                )
                for chunk in chunked:
                    if chunk.shape[0] >= min_length:
                        trajectories.append(chunk)
            elif traj_length >= min_length:
                trajectories.append(trajectory)

        # not necessary.
        # trajectories.sort(key=lambda t: t.shape[0])

        masked_trajectories = [
            torch.cat([trajectory, torch.ones((trajectory.shape[0], 1))], dim=1)
            for trajectory in trajectories
        ]
        self.dataset = GeoLifeDataset(masked_trajectories)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_length = int(len(self.dataset) * self.dm_config.ratio[0])
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_length, len(self.dataset) - train_length]
            )
        elif stage == "test":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.dm_config.batch_size,
            shuffle=True,
            num_workers=self.dm_config.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.dm_config.batch_size,
            shuffle=False,
            num_workers=self.dm_config.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    @staticmethod
    def collate_fn(batch):
        """_summary_

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        # batch: [B; (traj_length, 4)] (in order)

        # padded_trajectories: (traj_length, B, 4)
        padded_batch = torch.nn.utils.rnn.pad_sequence(batch)
        # padded_trajectories: (B, traj_length, 4)
        padded_batch = torch.permute(padded_batch, (1, 0, 2))

        # splited_batch: [4; (B, traj_length, 1)]
        splited_batch = torch.split(padded_batch, 1, dim=-1)

        real_batch = [
            torch.cat(splited_batch[0:2], dim=-1),  # (B, traj_length, 2)
            splited_batch[2],  # (B, traj_length, 1)
            splited_batch[3],  # (B, traj_length, 1)
            splited_batch[4],  # (B, traj_length, 1)
        ]
        return real_batch


class GeoIndDataModule(pl.LightningDataModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """

    def __init__(self, dm_config: DataModuleConfig):
        super().__init__()

        self.dm_config = dm_config
        self.dataset: GeoIndDataset
        self.train_dataset: Subset
        self.val_dataset: Subset

    @staticmethod
    def process(trajectory_files, min_length, max_length):
        "_summary_"

        trajectories: List[torch.Tensor] = []
        for file in trajectory_files:
            traj_csv = np.loadtxt(file, delimiter=",")
            # shape: (traj_length, 4)
            trajectory = torch.tensor(traj_csv, dtype=torch.float)
            if torch.isnan(trajectory).any():
                continue
            traj_length = trajectory.shape[0]
            if traj_length > max_length:
                chunked = torch.chunk(
                    trajectory, (traj_length // max_length) + 1, dim=0
                )
                for chunk in chunked:
                    if chunk.shape[0] >= min_length:
                        trajectories.append(chunk)
            elif traj_length >= min_length:
                trajectories.append(trajectory)
        return trajectories

    def prepare_data(self):
        min_length = self.dm_config.length_range[0]
        max_length = self.dm_config.length_range[1]

        trajectory_files = glob.glob(
            os.path.join(self.dm_config.data_dir, "**/*.csv"), recursive=True
        )
        real = GeoIndDataModule.process(trajectory_files, min_length, max_length)

        trajectory_files = glob.glob(
            os.path.join(self.dm_config.ind_dir, "**/*.csv"), recursive=True
        )
        synthetic = GeoIndDataModule.process(trajectory_files, min_length, max_length)

        # not necessary.
        # trajectories.sort(key=lambda t: t.shape[0])

        masked_real = [
            torch.cat([trajectory, torch.ones((trajectory.shape[0], 1))], dim=1)
            for trajectory in real
        ]
        masked_synthetic = [
            torch.cat([trajectory, torch.ones((trajectory.shape[0], 1))], dim=1)
            for trajectory in synthetic
        ]
        self.dataset = GeoIndDataset(masked_real, masked_synthetic)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_length = int(len(self.dataset) * self.dm_config.ratio[0])
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_length, len(self.dataset) - train_length]
            )
        elif stage == "test":
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.dm_config.batch_size,
            shuffle=True,
            num_workers=self.dm_config.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.dm_config.batch_size,
            shuffle=False,
            num_workers=self.dm_config.num_workers,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    @staticmethod
    def collate_fn(batch):
        """_summary_

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        # batch: [B; Tuple((traj_length, 4), (traj_length, 4))] (in order)

        real = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch])
        synthetic = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch])
        # padded_trajectories: (B, traj_length, 4)
        real = torch.permute(real, (1, 0, 2))
        synthetic = torch.permute(synthetic, (1, 0, 2))

        # splited_batch: [4; (B, traj_length, 1)]
        splited_real = torch.split(real, 1, dim=-1)
        splited_synthetic = torch.split(synthetic, 1, dim=-1)

        real_batch = (
            [
                torch.cat(splited_real[0:2], dim=-1),  # (B, traj_length, 2)
                splited_real[2],  # (B, traj_length, 1)
                splited_real[3],  # (B, traj_length, 1)
                splited_real[4],  # (B, traj_length, 1)
            ],
            [
                torch.cat(splited_synthetic[0:2], dim=-1),  # (B, traj_length, 2)
                splited_synthetic[2],  # (B, traj_length, 1)
                splited_synthetic[3],  # (B, traj_length, 1)
                splited_synthetic[4],  # (B, traj_length, 1)
            ],
        )
        return real_batch
