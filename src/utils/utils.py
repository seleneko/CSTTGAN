"""
## Information

- File Name: utils.py
- Author: Selene
- Date of Creation: 2023.03.20
- Date of Last Modification: 2023.05.17 (TODO: Update this)
- Python Version: 3.9.13
- License: GNU GPL v3.0
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as D
import torchvision.transforms as T
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from torch import Tensor, nn


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, inputs):
        return self.func(inputs)


class Utils:
    """_summary_

    Returns:
        _type_: _description_
    """

    @staticmethod
    def noise(
        size: torch.Size,
        max_norm: float = 1.0,
        mean: float = 0.0,
        noise_type: str = "gaussian",
    ) -> Tensor:
        """_summary_

        Args:
            size (torch.Size): _description_
            max_norm (float, optional): _description_. Defaults to 1.0.
            mean (float, optional): _description_. Defaults to 0.0.
            noise_type (str, optional): _description_. Defaults to "gaussian".

        Returns:
            Tensor: _description_
        """
        scale = torch.full(size, max_norm, dtype=torch.float32)

        if noise_type == "gaussian":
            dist = D.normal.Normal(mean, scale)
        elif noise_type == "laplacian":
            dist = D.laplace.Laplace(mean, scale)
        elif noise_type == "exponential":
            dist = D.exponential.Exponential(1 / scale)
        else:
            dist = D.normal.Normal(mean, scale)

        noise = dist.sample()
        return noise

    @staticmethod
    def trajectory_img(batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """_summary_

        Args:
            batch (torch.Tensor): _description_
            batch_idx (int): _description_

        Returns:
            torch.Tensor: _description_
        """

        tensor = []

        for idx, data in enumerate(batch):
            data = np.array(data)
            latitude, longitude, timestamp = data[:, 0], data[:, 1], data[:, 2]

            fig = plt.figure(figsize=(5, 5))
            axes: Axes3D = fig.add_axes([0, 0, 1, 1], projection="3d")
            axes.plot(latitude, longitude, timestamp, linewidth=1)
            axes.view_init(elev=20.0, azim=45)
            fpath = f"tmp/trajectory_{batch_idx}_{idx}.png"
            plt.savefig(fpath, dpi=300, bbox_inches="tight")
            img = Image.open(fpath)
            img.load()
            transfrom = T.Compose([T.Resize((1024, 1024)), T.PILToTensor()])
            tensor.append(transfrom(img))

        return torch.stack(tensor)

    @staticmethod
    def deepcopy(module: nn.Module, n_times: int) -> nn.ModuleList:
        """_summary_

        Args:
            module (nn.Module): _description_
            n_times (int): _description_

        Returns:
            nn.ModuleList: _description_
        """
        return nn.ModuleList([copy.deepcopy(module) for _ in range(n_times)])


class NonPositiveIntegerError(Exception):
    """_summary_"""

    def __init__(self, key: str, value: int):
        message = f"{key} must be a positive integer, but got {value}."
        super().__init__(message)


class NonPositiveFloatError(Exception):
    """_summary_"""

    def __init__(self, key: str, value: float):
        message = f"{key} must be a positive float, but got {value}."
        super().__init__(message)


class ListLengthError(Exception):
    """_summary_"""

    def __init__(self, name: str, expect: int, got: int):
        message = f"{name} must be a list with length {expect}, but got {got}."
        super().__init__(message)


class MissingKeyError(Exception):
    """_summary_"""

    def __init__(self, name: str, expect: str):
        message = f"Missing key in {name}, expect {expect}."
        super().__init__(message)
