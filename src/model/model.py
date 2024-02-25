"""
# model

This module contains the `CSTGAN` class, which is a PyTorch Lightning module that
implements  the CSTGAN model. This class is responsible for training the generator and
discriminator, and for generating fake trajectory sequences.

The model is composed of two parts: a generator and a discriminator.

## Classes

- `Generator`: generates fake trajectory sequences based on a given set of input tensors.
- `Discriminator`: classifies a given trajectory sequence as either real or fake.
- `CSTGAN`: a PyTorch Lightning module that implements the CSTGAN model.

## Information

- File Name: model.py
- Author: Selene
- Date of Creation: 2023.03.20
- Date of Last Modification: 2023.05.17 (TODO: Update this)
- Python Version: 3.9.13
- License: GNU GPL v3.0
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from model.losses import Accuracy, BCELoss, TrajLoss
from model.sublayers import (
    ClassificationLayer,
    EmbeddingLayer,
    FeatureFusionLayer,
    LSTMModelingLayer,
    RegressionLayer,
    SublayersConfig,
    TransformerModelingLayer,
)
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from utils.utils import Utils


class Generator(pl.LightningModule):
    """
    # Generator

    `Generator` is a PyTorch Lightning module that generates fake trajectory sequences based on a
    given set of input tensors. It is composed of the following sublayers:

    - `EmbeddingLayer`: embeds the input tensor into a sequence of dense vectors.
    - `FeatureFusionLayer`: combines the embedded tensor with noise vectors to create a trajectory
    for the temporal extraction layer to follow.
    - `LSTMModelingLayer`: processes the trajectory with an LSTM to get the hidden state of the
    sequence.
    - `TransformerModelingLayer`: processes the trajectory with an convolutional sparse transformer
    to get the temporal feature of the sequence.
    - `RegressionLayer`: converts the output of the temporal extraction layer into a tensor with the
    same shape as the input tensor.

    ## Inputs and Outputs

    `Generator` takes a dictionary of input tensors and returns a dictionary of output tensors. The
    input dictionary should contain a single key-value pair, where the key is a string and the value
    is a PyTorch tensor of shape `(batch_size, max_length, vocab_size)`. The output dictionary
    contains one key-value pair for each possible value of the input tensor. Each key is a string
    representing a possible value of the input tensor, and the corresponding value is a PyTorch
    tensor of the same shape as the input tensor.

    ## Parameters

    `Generator` has two parameters:

    - `vocab_size` (Dict[str, int]): a dictionary containing the number of unique elements in each
    category.
    - `config` (SublayersConfig): an object containing the configuration information for this layer.
    """

    def __init__(self, vocab_size: Dict[str, int], config: SublayersConfig):
        super().__init__()

        self.keys = vocab_size.keys()

        # Sublayers
        self.embedding_layer = EmbeddingLayer(vocab_size, config)
        self.feature_fusion_layer = FeatureFusionLayer(vocab_size, config)
        self.temporal_extraction_layer = (
            TransformerModelingLayer(config)
            if config.use_transformer["generator"]
            else LSTMModelingLayer(config)
        )
        self.regression_layer = RegressionLayer(vocab_size, config)

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def _forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # <=| inputs: Dict[str, Tensor (B, max_length, vocab_size)] (real)

        # I. Embedding Layer
        # <- inputs: Dict[str, Tensor (B, max_length, vocab_size)]
        # -> input_sequence: [N+1; (B, max_length, fc_out)]
        # -> embeddings: [N-1(mask); (B, max_length, fc_out)]
        embeddings = self.embedding_layer(inputs)

        # II. Feature Fusion Layer
        # <- embeddings: [N-1; (B, max_length, fc_out)]
        # <- noise: (B, noise_dim)
        # -> emb_traj: (B, max_length, features)
        emb_traj = self.feature_fusion_layer(embeddings, inputs["noise"])

        # III. Temporal Extraction Layer
        # <- emb_traj: (B, max_length, features)
        # -> temporal: (B, max_length, extraction_dim)
        temporal, _ = self.temporal_extraction_layer(emb_traj)

        # IV. Regression Layer
        # <- temporal: (B, max_length, extraction_dim)
        # <- mask: (B, max_length)
        # -> outputs: [N; (B, max_length, vocab_size)]
        outputs = self.regression_layer(temporal, inputs["mask"])

        # |=> return: Dict[str, Tensor (B, max_length, vocab_size)] (fake)
        return dict(zip(self.keys, outputs))


class Discriminator(pl.LightningModule):
    """
    # Discriminator

    `Discriminator` is a PyTorch Lightning module for training a generative model with adversarial
    training. It takes in a dictionary of input tensors and returns a single tensor representing the
    probability that the input is real (i.e., drawn from the true data distribution) or fake (i.e.,
    generated by the generator model). It is composed of the following sublayers:

    - `EmbeddingLayer`: embeds the input tensor into a sequence of dense vectors.
    - `FeatureFusionLayer`: combines the embedded tensor with noise vectors to create a trajectory
    for the temporal extraction layer to follow.
    - `LSTMModelingLayer`: processes the trajectory with an LSTM to get the cell state at the end of
    the sequence.
    - `TransformerModelingLayer`: processes the trajectory with an convolutional sparse transformer
    to get the temporal feature of the sequence.
    - `ClassificationLayer`: converts the output of the temporal extraction output into a single
    value representing the probability that the input is real or fake.

    Note that the discriminator uses the same `EmbeddingLayer`, `FeatureFusionLayer`, and
    `LSTMModelingLayer` or `TransformerModelingLayer`, as the generator, but with different
    parameters. The `classification_layer` is a fully connected layer that maps the temporal
    extraction output to a scalar representing the probability that the input is real.

    ## Inputs and Outputs

    `Discriminator` takes a dictionary of inputs, which can be either real or fake data. It returns
    a tensor of shape `(B, 1)`, which represents the probability that the input is real. The output
    value is in the range `[0, 1]`, where values close to `1` indicate that the input is likely to
    be real and values close to `0` indicate that the input is likely to be fake.

    ## Parameters

    `Discriminator` has two parameters:

    - `vocab_size` (Dict[str, int]): a dictionary containing the number of unique elements in each
    category.
    - `config` (SublayersConfig): an object containing the configuration information for this layer.
    """

    def __init__(self, vocab_size: Dict[str, int], config: SublayersConfig):
        super().__init__()

        # Sublayers
        self.embedding_layer = EmbeddingLayer(vocab_size, config)
        self.feature_fusion_layer = FeatureFusionLayer(vocab_size, config)
        self.temporal_extraction_layer = (
            TransformerModelingLayer(config)
            if config.use_transformer["discriminator"]
            else LSTMModelingLayer(config)
        )
        self.classification_layer = ClassificationLayer(config)

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def _forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        # <=| inputs: Dict[str, Tensor (B, max_length, vocab_size)] (real? fake?)

        # I. Embedding Layer
        # <- inputs: Dict[str, Tensor (B, max_length, vocab_size)]
        # -> embeddings: [N; (B, max_length, fc_out)]
        embeddings = self.embedding_layer(inputs)

        # II. Feature Fusion Layer
        # <- embeddings: [N; (B, max_length, fc_out)]
        # -> emb_traj: (B, max_length, features)
        emb_traj = self.feature_fusion_layer(embeddings)

        # III. Temporal Extraction Layer
        # <- emb_traj: (B, max_length, features)
        # -> temporal: (B, extraction_dim)
        _, temporal = self.temporal_extraction_layer(emb_traj)

        # IV. Output Layer
        # <- temporal: (B, extraction_dim)
        # -> outputs: (B, 1)
        outputs = self.classification_layer(temporal)

        # |=> outputs: Tensor (B, 1)
        return outputs  # in [0, 1]


@dataclass
class OptimizersConfig:
    "Optimizers Config"

    learning_rate: float
    betas: List[float]


@dataclass
class Criterions:
    "Criterions"

    g_loss: nn.Module
    d_loss: nn.Module
    d_acc: nn.Module


@dataclass
class CriterionsConfig:
    "Criterions Config"

    alpha: float
    beta: float
    gamma: float


class CSTGANPreserve(pl.LightningModule):
    """
    # CSTGANPreserve

    `CSTGAN` is a PyTorch Lightning module that implements a Convolutional Sparse Transformer based
    Generative Adversarial Network (GAN) for generating synthetic trajectories.

    The model uses a `Generator` and a `Discriminator` to train the model. The `Generator` generates
    synthetic trajectories from real trajectories, while the `Discriminator` discriminates between
    real and synthetic trajectories.
    """

    def __init__(
        self,
        model_config: SublayersConfig,
        optimizers: OptimizersConfig,
        criterions: CriterionsConfig,
    ):
        super().__init__()

        vocab_size = {"latlng": 2, "alt": 1, "time": 1, "mask": 1}
        self.keys = vocab_size.keys()
        self.latent_dim = model_config.latent_dim
        self.optimizers_config = optimizers

        # Build the trajectory generator
        self.generator = Generator(vocab_size, model_config)

        # Build the trajectory discriminator
        self.discriminator = Discriminator(vocab_size, model_config)
        self.discriminator.train(False)

        self.criterions = Criterions(
            g_loss=TrajLoss(
                alpha=criterions.alpha,
                beta=criterions.beta,
                gamma=criterions.gamma,
            ),
            d_loss=BCELoss(),
            d_acc=Accuracy(),
        )
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

    @property
    def automatic_optimization(self) -> bool:
        """
        [Note] Since PyTorch Lightning 2.0.0, training with multiple optimizers is only supported
        with manual optimization. See
        https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        """
        return False

    @property
    def use_grad_scaler(self) -> bool:
        """
        Permets-tu?
        """
        return torch.cuda.is_available() and True

    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        optimizer = torch.optim.Adam  # ???
        betas = tuple(self.optimizers_config.betas)
        g_opt = optimizer(
            self.generator.parameters(),
            lr=self.optimizers_config.learning_rate,
            betas=betas,
        )
        d_opt = optimizer(
            self.discriminator.parameters(),
            lr=self.optimizers_config.learning_rate,
            betas=betas,
        )
        return g_opt, d_opt

    def _manual_opt(self, opt: Optimizer, loss: Tensor):
        if self.use_grad_scaler:
            opt.zero_grad()
            self.scaler.scale(loss).backward()  # type: ignore
            self.scaler.step(opt)
            self.scaler.update()
        else:
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

    def training_step(self, *args, **kwargs):
        return self._training_step(*args, **kwargs)

    def _training_step(self, batch: List[Tensor], _):
        # <=| batch: [len(keys); (B, max_length, vocab_size)]

        # Extract the batch size
        batch_size = batch[0].shape[0]

        # Extract the optimizers
        # pylint: disable=unpacking-non-sequence
        g_opt, d_opt = self.optimizers()  # type: ignore

        # Create dictionary of real trajectories
        # real_traj: Dict[str, Tensor (B, max_length, vocab_size)]
        # keys = ["latlng", "alt", "time", "mask"]
        real_traj: Dict[str, Tensor] = dict(zip(self.keys, batch))

        #######################
        # Train the generator #
        #######################
        self._train_the_generator(real_traj, batch_size, g_opt)

        ###########################
        # Train the discriminator #
        ###########################
        self._train_the_discriminator(real_traj, batch_size, d_opt)

    def _train_the_generator(self, real_traj, batch_size, g_opt):
        # Extract ground truths for real and synthetic trajectories
        real_gt = torch.ones((batch_size, 1), device=self.device)

        # Generate noise
        # noise: (B, latent_dim)
        noise_size = torch.Size([batch_size, self.latent_dim])
        real_traj["noise"] = torch.as_tensor(
            Utils.noise(noise_size), device=self.device
        )

        # Generate synthetic trajectories from real trajectories
        fake_traj: Dict[str, Tensor] = self.generator(real_traj)
        pred: Tensor = self.discriminator(fake_traj)

        # Compute the generator loss
        g_loss: Tensor = self.criterions.g_loss(real_traj, fake_traj, pred, real_gt)
        self._manual_opt(g_opt, g_loss)
        self.log("train/g_loss", g_loss, prog_bar=True)

    def _train_the_discriminator(self, real_traj, batch_size, d_opt):
        # Extract ground truths for real and synthetic trajectories
        real_gt = torch.ones((batch_size, 1), device=self.device)
        fake_gt = torch.zeros((batch_size, 1), device=self.device)

        # Generate noise
        noise_size = torch.Size([batch_size, self.latent_dim])
        real_traj["noise"] = torch.as_tensor(
            Utils.noise(noise_size), device=self.device
        )

        # Generate synthetic trajectories from real trajectories
        fake_traj: Dict[str, Tensor] = self.generator(real_traj)

        # Compute the discriminator loss
        real_loss: Tensor = self.criterions.d_loss(
            self.discriminator(real_traj), real_gt
        )
        fake_loss: Tensor = self.criterions.d_loss(
            self.discriminator(fake_traj), fake_gt
        )
        d_loss = torch.add(real_loss, fake_loss).div(2)
        self._manual_opt(d_opt, d_loss)
        self.log("train/d_loss", d_loss, prog_bar=True)

    def validation_step(self, *args, **kwargs):
        return self._validation_step(*args, **kwargs)

    def _validation_step(self, batch: List[Tensor], _):
        # <=| batch: [len(keys); (B, max_length, vocab_size)]

        # Extract the batch size
        batch_size = batch[0].shape[0]

        # Create dictionary of real trajectories
        # real_traj: Dict[str, Tensor (B, max_length, vocab_size)]
        # keys = ["latlng", "alt", "time", "mask"]
        real_traj: Dict[str, Tensor] = dict(zip(self.keys, batch))

        # Generate noise
        # noise: (B, latent_dim)
        noise_size = torch.Size([batch_size, self.latent_dim])
        real_traj["noise"] = torch.as_tensor(
            Utils.noise(noise_size), device=self.device
        )

        ##########################
        # Validate the generator #
        ##########################
        self._validate_the_generator(real_traj, batch_size)

        ##############################
        # Validate the discriminator #
        ##############################
        self._validate_the_discriminator(real_traj, batch_size)

    def _validate_the_generator(self, real_traj, batch_size):
        # Extract ground truths for real and synthetic trajectories
        real_gt = torch.ones((batch_size, 1), device=self.device)

        # Generate synthetic trajectories from real trajectories
        fake_traj: Dict[str, Tensor] = self.generator(real_traj)

        # Compute the generator loss
        pred: Tensor = self.discriminator(fake_traj)
        g_loss: Tensor = self.criterions.g_loss(real_traj, fake_traj, pred, real_gt)
        self.log("val/g_loss", g_loss)

    def _validate_the_discriminator(self, real_traj, batch_size):
        # Extract ground truths for real and synthetic trajectories
        real_gt = torch.ones((batch_size, 1), device=self.device)
        fake_gt = torch.zeros((batch_size, 1), device=self.device)

        # Generate synthetic trajectories from real trajectories
        fake_traj: Dict[str, Tensor] = self.generator(real_traj)

        # Compute the discriminator loss
        real_loss: Tensor = self.criterions.d_loss(
            self.discriminator(real_traj), real_gt
        )
        fake_loss: Tensor = self.criterions.d_loss(
            self.discriminator(fake_traj), fake_gt
        )
        d_loss = torch.add(real_loss, fake_loss).div(2)
        self.log("val/d_loss", d_loss)

        # Compute the discriminator accuracy
        real_acc: Tensor = self.criterions.d_acc(self.discriminator(real_traj), real_gt)
        fake_acc: Tensor = self.criterions.d_acc(self.discriminator(fake_traj), fake_gt)
        d_acc = torch.add(real_acc, fake_acc).div(2)
        self.log("val/d_acc", d_acc)

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def _forward(self, batch: List[Tensor]):
        # <=| batch: [len(keys); (B, max_length, vocab_size)]

        # Extract the batch size
        batch_size = batch[0].shape[0]

        # Create dictionary of real trajectories
        # real_traj: Dict[str, Tensor (B, max_length, vocab_size)]
        # keys = ["latlng", "alt", "time", "mask"]
        real_traj: Dict[str, Tensor] = dict(zip(self.keys, batch))

        # Generate noise
        # noise: (B, latent_dim)
        noise_size = torch.Size([batch_size, self.latent_dim])
        real_traj["noise"] = torch.as_tensor(
            Utils.noise(noise_size), device=self.device
        )

        fake_traj: Dict[str, Tensor] = self.generator(real_traj)

        for _ in range(20):
            fake_traj["noise"] = torch.as_tensor(
                Utils.noise(noise_size), device=self.device
            )
            fake_traj = self.generator(fake_traj)

        return fake_traj


class CSTGANAttack(pl.LightningModule):
    """
    # CSTGANAttack

    `CSTGAN` is a PyTorch Lightning module that implements a Convolutional Sparse Transformer based
    Generative Adversarial Network (GAN) for generating synthetic trajectories.

    The model uses a `Generator` and a `Discriminator` to train the model. The `Generator` generates
    synthetic trajectories from real trajectories, while the `Discriminator` discriminates between
    real and synthetic trajectories.
    """

    def __init__(
        self,
        model_config: SublayersConfig,
        optimizers: OptimizersConfig,
        criterions: CriterionsConfig,
    ):
        super().__init__()

        vocab_size = {"latlng": 2, "alt": 1, "time": 1, "mask": 1}
        self.keys = vocab_size.keys()
        self.latent_dim = model_config.latent_dim
        self.optimizers_config = optimizers

        # Build the trajectory generator
        self.generator = Generator(vocab_size, model_config)

        # Build the trajectory discriminator
        self.discriminator = Discriminator(vocab_size, model_config)
        self.discriminator.train(False)

        self.criterions = Criterions(
            g_loss=TrajLoss(
                alpha=criterions.alpha,
                beta=criterions.beta,
                gamma=criterions.gamma,
            ),
            d_loss=BCELoss(),
            d_acc=Accuracy(),
        )
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

    @property
    def automatic_optimization(self) -> bool:
        """
        [Note] Since PyTorch Lightning 2.0.0, training with multiple optimizers is only supported
        with manual optimization. See
        https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        """
        return False

    @property
    def use_grad_scaler(self) -> bool:
        """
        Permets-tu?
        """
        return torch.cuda.is_available() and True

    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        optimizer = torch.optim.Adam  # ???
        betas = tuple(self.optimizers_config.betas)
        g_opt = optimizer(
            self.generator.parameters(),
            lr=self.optimizers_config.learning_rate,
            betas=betas,
        )
        d_opt = optimizer(
            self.discriminator.parameters(),
            lr=self.optimizers_config.learning_rate,
            betas=betas,
        )
        return g_opt, d_opt

    def _manual_opt(self, opt: Optimizer, loss: Tensor):
        if self.use_grad_scaler:
            opt.zero_grad()
            self.scaler.scale(loss).backward()  # type: ignore
            self.scaler.step(opt)
            self.scaler.update()
        else:
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

    def training_step(self, *args, **kwargs):
        return self._training_step(*args, **kwargs)

    def _training_step(self, batch: List[Tensor], _):
        # <=| batch: Tuple([len(keys); (B, max_length, vocab_size)], ...)

        # Extract the batch
        real_batch, synthetic_batch = batch
        # Extract the batch size
        batch_size = real_batch[0].shape[0]

        # Extract the optimizers
        # pylint: disable=unpacking-non-sequence
        g_opt, d_opt = self.optimizers()  # type: ignore

        # Create dictionary of real trajectories
        # real_traj: Dict[str, Tensor (B, max_length, vocab_size)]
        # keys = ["latlng", "alt", "time", "mask"]
        real_traj: Dict[str, Tensor] = dict(zip(self.keys, real_batch))
        synthetic_traj: Dict[str, Tensor] = dict(zip(self.keys, synthetic_batch))

        #######################
        # Train the generator #
        #######################
        self._train_the_generator(real_traj, synthetic_traj, batch_size, g_opt)

        ###########################
        # Train the discriminator #
        ###########################
        self._train_the_discriminator(real_traj, synthetic_traj, batch_size, d_opt)

    def _train_the_generator(self, real_traj, synthetic_traj, batch_size, g_opt):
        # Extract ground truths for real and attack trajectories
        real_gt = torch.ones((batch_size, 1), device=self.device)

        # Generate noise
        # noise: (B, latent_dim)
        noise_size = torch.Size([batch_size, self.latent_dim])
        synthetic_traj["noise"] = torch.as_tensor(
            Utils.noise(noise_size), device=self.device
        )

        # Generate attack trajectories from synthetic trajectories
        attack_traj: Dict[str, Tensor] = self.generator(synthetic_traj)
        pred: Tensor = self.discriminator(attack_traj)

        # Compute the generator loss
        g_loss: Tensor = self.criterions.g_loss(real_traj, attack_traj, pred, real_gt)
        self._manual_opt(g_opt, g_loss)
        self.log("train/g_loss", g_loss, prog_bar=True)

    def _train_the_discriminator(self, real_traj, synthetic_traj, batch_size, d_opt):
        # Extract ground truths for real and attack trajectories
        real_gt = torch.ones((batch_size, 1), device=self.device)
        fake_gt = torch.zeros((batch_size, 1), device=self.device)

        # Generate noise
        noise_size = torch.Size([batch_size, self.latent_dim])
        synthetic_traj["noise"] = torch.as_tensor(
            Utils.noise(noise_size), device=self.device
        )

        # Generate attack trajectories from synthetic trajectories
        attack_traj: Dict[str, Tensor] = self.generator(synthetic_traj)

        # Compute the discriminator loss
        real_loss: Tensor = self.criterions.d_loss(
            self.discriminator(real_traj), real_gt
        )
        fake_loss: Tensor = self.criterions.d_loss(
            self.discriminator(attack_traj), fake_gt
        )
        d_loss = torch.add(real_loss, fake_loss).div(2)
        self._manual_opt(d_opt, d_loss)
        self.log("train/d_loss", d_loss, prog_bar=True)

    def validation_step(self, *args, **kwargs):
        return self._validation_step(*args, **kwargs)

    def _validation_step(self, batch: List[Tensor], _):
        # <=| batch: Tuple([len(keys); (B, max_length, vocab_size)], ...)

        # Extract the batch
        real_batch, synthetic_batch = batch
        # Extract the batch size
        batch_size = real_batch[0].shape[0]

        # Create dictionary of real trajectories
        # real_traj: Dict[str, Tensor (B, max_length, vocab_size)]
        # keys = ["latlng", "alt", "time", "mask"]
        real_traj: Dict[str, Tensor] = dict(zip(self.keys, real_batch))
        synthetic_traj: Dict[str, Tensor] = dict(zip(self.keys, synthetic_batch))

        # Generate noise
        # noise: (B, latent_dim)
        noise_size = torch.Size([batch_size, self.latent_dim])
        synthetic_traj["noise"] = torch.as_tensor(
            Utils.noise(noise_size), device=self.device
        )

        ##########################
        # Validate the generator #
        ##########################
        self._validate_the_generator(real_traj, synthetic_traj, batch_size)

        ##############################
        # Validate the discriminator #
        ##############################
        self._validate_the_discriminator(real_traj, synthetic_traj, batch_size)

    def _validate_the_generator(self, real_traj, synthetic_traj, batch_size):
        # Extract ground truths for real and synthetic trajectories
        real_gt = torch.ones((batch_size, 1), device=self.device)

        # Generate attack trajectories from synthetic trajectories
        attack_traj: Dict[str, Tensor] = self.generator(synthetic_traj)

        # Compute the generator loss
        pred: Tensor = self.discriminator(attack_traj)
        g_loss: Tensor = self.criterions.g_loss(real_traj, attack_traj, pred, real_gt)
        self.log("val/g_loss", g_loss)

    def _validate_the_discriminator(self, real_traj, synthetic_traj, batch_size):
        # Extract ground truths for real and synthetic trajectories
        real_gt = torch.ones((batch_size, 1), device=self.device)
        fake_gt = torch.zeros((batch_size, 1), device=self.device)

        # Generate attack trajectories from synthetic trajectories
        attack_traj: Dict[str, Tensor] = self.generator(synthetic_traj)

        # Compute the discriminator loss
        real_loss: Tensor = self.criterions.d_loss(
            self.discriminator(real_traj), real_gt
        )
        fake_loss: Tensor = self.criterions.d_loss(
            self.discriminator(attack_traj), fake_gt
        )
        d_loss = torch.add(real_loss, fake_loss).div(2)
        self.log("val/d_loss", d_loss)

        # Compute the discriminator accuracy
        real_acc: Tensor = self.criterions.d_acc(self.discriminator(real_traj), real_gt)
        fake_acc: Tensor = self.criterions.d_acc(
            self.discriminator(attack_traj), fake_gt
        )
        d_acc = torch.add(real_acc, fake_acc).div(2)
        self.log("val/d_acc", d_acc)

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def _forward(self, batch: List[Tensor]):
        # <=| batch: [len(keys); (B, max_length, vocab_size)]

        # Extract the batch size
        batch_size = batch[0].shape[0]

        # Create dictionary of real trajectories
        # real_traj: Dict[str, Tensor (B, max_length, vocab_size)]
        # keys = ["latlng", "alt", "time", "mask"]
        preserved_traj: Dict[str, Tensor] = dict(zip(self.keys, batch))

        # Generate noise
        # noise: (B, latent_dim)
        noise_size = torch.Size([batch_size, self.latent_dim])
        preserved_traj["noise"] = torch.as_tensor(
            Utils.noise(noise_size), device=self.device
        )

        attack_traj: Dict[str, Tensor] = self.generator(preserved_traj)
        for _ in range(2):
            attack_traj["noise"] = torch.as_tensor(
                Utils.noise(noise_size), device=self.device
            )
            attack_traj: Dict[str, Tensor] = self.generator(attack_traj)

        return attack_traj
