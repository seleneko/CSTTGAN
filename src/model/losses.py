"""
# model

This file contains three PyTorch modules used in the GAN.

## Classes

- `BCELoss`: implements the binary cross-entropy loss function.
- `TrajLoss`: implements a custom loss function for the generator.
- `Accuracy`: implements a custom accuracy metric for the discriminator.

## Information

- File Name: model.py
- Author: Selene
- Date of Creation: 2023.03.21
- Date of Last Modification: 2023.05.17 (TODO: Update this)
- Python Version: 3.9.13
- License: GNU GPL v3.0
"""

from typing import Dict, List

import operator
import torch
from torch import Tensor, nn


class BCELoss(nn.Module):
    """
    # BCELoss

    `BCELoss` is a PyTorch module for implementing binary cross-entropy loss. It is designed to be
    used as the loss function for the discriminator. The class inherits from the `nn.Module` class
    and utilizes the `nn.BCELoss()` function to calculate the loss.
    """

    def __init__(self):
        super().__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """_summary_

        Args:
            y_pred (Tensor): _description_
            y_true (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        return self.bce_criterion(y_pred, y_true)


class TrajLoss(nn.Module):
    """
    # TrajLoss

    `TrajLoss` is a PyTorch module that computes a custom loss function for the generator. The loss
    function consists of several components, including binary cross-entropy loss, masked mean
    squared error loss, and categorical cross-entropy loss.

    ## Forward
    `forward` Computes the loss function given the real and fake trajectories, predicted and true
    labels, and returns the loss tensor. The method takes the following arguments:

    - `real_traj`: A dictionary containing tensors representing the real trajectory.
    - `fake_traj`: A dictionary containing tensors representing the fake trajectory.
    - `y_pred`: A tensor of shape `(batch_size, 1)` representing the predicted labels.
    - `y_true`: A tensor of shape `(batch_size, 1)` representing the true labels.

    ## Loss

    The method computes the following components of the loss:

    - Binary cross-entropy loss between the predicted and true labels.
    - Masked mean squared error loss between the fake and real trajectory latitudes and longitudes.
    - Categorical cross-entropy loss between the fake and real trajectory categories, days, and
    hours, weighted by the mask.
    - The method returns the sum of these loss components, weighted by predetermined coefficients
    and normalized by the batch size.
    """

    def __init__(self, alpha: float, beta: float, gamma: float):
        super().__init__()

        self.bce_criterion = nn.BCELoss()
        self.sce_criterion = nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        real: Dict[str, Tensor],
        fake: Dict[str, Tensor],
        y_pred: Tensor,
        y_true: Tensor,
    ) -> Tensor:
        """_summary_

        Args:
            real (Dict[str, Tensor]): _description_
            fake (Dict[str, Tensor]): _description_
            y_pred (Tensor): _description_
            y_true (Tensor): _description_

        Returns:
            Tensor: _description_
        """

        batch_size = real["latlng"].shape[0]

        # not good
        real_traj: List[Tensor] = [
            real["latlng"],
            real["alt"],
            real["time"],
            real["mask"],
        ]
        fake_traj: List[Tensor] = [
            fake["latlng"],
            fake["alt"],
            fake["time"],
            fake["mask"],
        ]

        traj_length = real_traj[3].clone().detach().sum(dim=1)

        bce_loss: Tensor = self.bce_criterion(y_pred, y_true)  # BCE(y^r, y^p)

        masked_latlon_full = (
            (fake_traj[0].sub(real_traj[0]))
            .mul(fake_traj[0].sub(real_traj[0]))
            .mul(torch.cat([real_traj[3] for _ in range(2)], dim=2))
            .sum(dim=1)
            .sum(dim=1, keepdim=True)
        )

        masked_latlon_mse = masked_latlon_full.div(traj_length).sum()

        result = (bce_loss.mul(self.alpha).add(masked_latlon_mse.mul(self.beta))).div(
            batch_size
        )

        return result


class Accuracy(nn.Module):
    """
    `Accuracy` is a PyTorch module for calculating the accuracy of a binary classifier. It is
    designed to be used as the accuracy function for the discriminator. The class inherits from the
    `nn.Module` class.
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """_summary_

        Args:
            y_pred (Tensor): _description_
            y_true (Tensor): _description_

        Returns:
            Tensor: _description_
        """

        ops = operator.gt if y_true[0] == 1 else operator.lt
        return torch.div(sum(int(ops(y, 0.5)) for y in y_pred), y_true.shape[0])
