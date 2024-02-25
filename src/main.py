#!/usr/bin/python3
"""main.py"""

import torch
from data.dataloader import GeoIndDataModule, GeoLifeDataModule
from model.model import CSTGANAttack, CSTGANPreserve
from pytorch_lightning.cli import LightningCLI

# #.*\.\n

__author__ = "Selene"
__copyright__ = None  # GNU GPL is a copyleft license.
__license__ = "GNU General Public License v3.0"
__version__ = "1.5.1"
__maintainer__ = "Selene"
__email__ = "selenotoxo@pm.me"
__status__ = "Production"


class CLI(LightningCLI):
    """_summary_

    Args:
        LightningCLI (_type_): _description_
    """

    def fit(self, model, **kwargs):
        """_summary_

        Args:
            model (_type_): _description_
        """
        model = torch.compile(model)
        self.trainer.fit(model, **kwargs)  # type: ignore


def cli_main(task: str = "preserve"):
    """_summary_"""
    torch.set_float32_matmul_precision("high")
    model_class, datamodule_class = (
        (CSTGANPreserve, GeoLifeDataModule)
        if task == "preserve"
        else (CSTGANAttack, GeoIndDataModule)
    )
    LightningCLI(
        model_class=model_class,
        datamodule_class=datamodule_class,
        save_config_callback=None,
    )


if __name__ == "__main__":
    cli_main(task="preserve")
