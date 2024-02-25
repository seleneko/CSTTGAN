"""
# sublayers

This module contains helper classes for the `CSTGAN` model. The classes in this module are not meant
to be used directly, but are instead used by the `CSTGAN` model.

## Classes

- `FCLayer`: A helper class to create a fully connected layer.
- `EmbeddingLayer`: A helper class to create an embedding layer.
- `FeatureFusionLayer`: A helper class to create a feature fusion layer.
- `ConvTransModelingLayer`: A helper class to create a convolutional transformer modeling layer.
- `TransformerModelingLayer`: A helper class to create a transformer modeling layer.
- `LSTMModelingLayer`: A helper class to create an LSTM modeling layer.
- `RegressionLayer`: A helper class to create a regression layer.

## Information

- File Name: sublayers.py
- Author: Selene
- Date of Creation: 2023.03.25
- Date of Last Modification: 2023.05.17 (TODO: Update this)
- Python Version: 3.9.13
- License: GNU GPL v3.0
"""


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from utils.utils import (
    ListLengthError,
    NonPositiveFloatError,
    NonPositiveIntegerError,
    MissingKeyError,
)
from model.transformer import (
    ContextEmbedding,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerConfig,
)


@dataclass
class SublayersConfig:
    "Sublayers Config"

    fc_out: Dict[str, int]
    latent_dim: int
    scale_factor: float
    features: int
    extraction_dim: int
    use_transformer: Dict[str, bool]
    transformer: TransformerConfig


class FCLayer(nn.Module):
    """
    # FC Layer

    `FCLayer` is a helper class to create a fully connected layer in a neural network. It
    initializes a linear transformation layer using PyTorch's `nn.Linear` class, with weights
    initialized using the Kaiming uniform initialization and biases set to zero. The activation
    function used can be specified using the `activation` parameter during initialization, and the
    available options are ReLU, sigmoid, and tanh.

    ## Parameters

    `FCLayer` has three parameters:

    - `in_features` (int): the number of input features
    - `out_features` (int): the number of output features
    - `activation` (str, optional): the activation function to use. Possible options are "relu",
    "sigmoid", and "tanh". Defaults to "relu".

    ## Input

    `FCLayer` takes in a PyTorch tensor of shape `(batch_size, in_features)` as input.

    ## Output

    `FCLayer` returns a PyTorch tensor of shape `(batch_size, out_features)` as output, which is the
    result of applying the activation function to the linear transformation of the input.

    Note that the weights and biases of the linear transformation layer are initialized during the
    class's initialization and cannot be directly accessed or modified.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
    ):
        super().__init__()

        if activation not in ["relu", "sigmoid", "tanh"]:
            raise ValueError(
                (
                    f"Invalid value for activation: {activation}. ",
                    "Possible options are 'relu', 'sigmoid', and 'tanh'.",
                )
            )
        if in_features <= 0:
            raise NonPositiveIntegerError("in_features", in_features)
        if out_features <= 0:
            raise NonPositiveIntegerError("out_features", out_features)

        fc_layer = nn.Linear(in_features, out_features)
        nn.init.kaiming_uniform_(fc_layer.weight, nonlinearity=activation)
        nn.init.zeros_(fc_layer.bias)

        activation_fn: nn.Module = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }.get(activation, None)

        self.model = nn.Sequential(fc_layer, activation_fn)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        `forward` performs the forward pass of the layer. It takes in a PyTorch tensor of shape
        `(batch_size, in_features)` as input, and returns a PyTorch tensor of shape
        `(batch_size, out_features)` as output. The output is obtained by applying the activation
        function to the linear transformation of the input.
        """

        return self.model(inputs)


class EmbeddingLayer(nn.Module):
    """
    # Embedding Layer

    `EmbeddingLayer` implements an embedding layer that takes in a dictionary of input tensors with
    size (batch_size, max_length, vocab_size) for both generator and discriminator and returns a
    list of embedded tensors with size (batch_size, max_length, fc_out).

    ## Parameters

    `EmbeddingLayer` has two parameters:

    - `vocab_size` (Dict[str, int]): a dictionary containing the number of unique elements in each
    category.
    - `config` (SublayersConfig): an object containing the configuration information for this layer.

    ## Input

    `EmbeddingLayer` takes in a dictionary of input tensors:

    - `inputs` (Dict[str, Tensor]): a dictionary of input tensors with size
    (batch_size, max_length, vocab_size).

    ## Output

    `EmbeddingLayer` returns a list of embedded tensors:

    - `embeddings` (List[Tensor]): a list of `N-1` tensors of shape
    `(batch_size, max_length, fc_out)` that represents the embedded input categories, where `N` is
    the number of input categories.
    """

    def __init__(self, vocab_size: Dict[str, int], config: SublayersConfig):
        super().__init__()

        self.keys = vocab_size.keys()

        if len(config.fc_out) != 2:
            raise ListLengthError("fc_out", expect=2, got=len(config.fc_out))
        if "latlng" not in config.fc_out or "others" not in config.fc_out:
            raise MissingKeyError("fc_out", expect="latlng, others")

        fc_out_latlng, fc_out_others = config.fc_out["latlng"], config.fc_out["others"]

        if fc_out_latlng <= 0:
            raise NonPositiveIntegerError("fc_out['latlng']", fc_out_latlng)
        if fc_out_others <= 0:
            raise NonPositiveIntegerError("fc_out['others']", fc_out_others)

        self.embedding_mlps = nn.ModuleDict(
            {  # FC: (vocab_size) -> (fc_out)
                key: FCLayer(
                    in_features=vocab_size[key],
                    out_features=fc_out_latlng if key == "latlng" else fc_out_others,
                    activation="relu",
                )
                for key in self.keys
                if key != "mask"
            }
        )

    def forward(self, inputs: Dict[str, Tensor]) -> List[Tensor]:
        """
        `forward` performs the forward pass of the `EmbeddingLayer`. It takes in a dictionary of
        tensors that includes the input categories. For each input category, it first applies a
        fully-connected layer with ReLU activation to embed the input category into a
        low-dimensional space. The resulting embeddings are stacked along the sequence length
        dimension to form a tensor of shape `(batch_size, max_length, fc_out)`. The embeddings of
        all input categories, except for the mask tensor, are collected in a list and returned as
        the first element of the output tuple.
        """

        # <- inputs: Dict[str, Tensor (B, max_length, vocab_size)]

        # List of embeddings
        embeddings: List[Tensor] = []

        for key in self.keys:
            # Do not embed mask
            if key == "mask":
                continue

            # inputs_data: (B, max_length, vocab_size)
            input_data = inputs[key]

            # unbinded: (max_length, B, vocab_size)
            permuted = input_data.permute(1, 0, 2)

            # embedded: [max_length; (B, fc_out)]
            embedded = [self.embedding_mlps[key](x) for x in permuted]

            # stacked: (B, max_length, fc_out)
            stacked = torch.stack(embedded, dim=1)

            embeddings.append(stacked)

        # -> embeddings: [N-1; (B, max_length, fc_out)]
        return embeddings


class FeatureFusionLayer(nn.Module):
    """
    # Feature Fusion Layer

    `FeatureFusionLayer` implements a feature fusion layer for both generator and discriminator.
    This layer is responsible for combining input embeddings and noise vectors into a fused feature
    representation that can be used as input to subsequent layers. It takes in a vocabulary size
    dictionary and a configuration object. The output of this layer is a tensor that has undergone
    feature fusion.

    ## Parameters

    `FeatureFusionLayer` has two parameters:

    - `vocab_size` (Dict[str, int]): a dictionary containing the number of unique elements in each
    category.
    - `config` (SublayersConfig): an object containing the configuration information for this layer.

    ## Input

    `FeatureFusionLayer` takes in two inputs:

    - `embeddings` (List[Tensor]): a list of tensors, each tensor represents the output of a
    previous layer and has shape `(batch_size, max_length, fc_out)`, where `max_length` is the
    maximum sequence length, and `fc_out` is the output size of the previous layer. The `embeddings`
    list has length `N-1`, where `N` is the number of input categories.
    - `noise` (Optional[Tensor]): a tensor representing noise that can be concatenated with the
    embeddings. It has shape `(batch_size, latent_dim)` where `latent_dim` is a hyperparameter.

    ## Output

    `FeatureFusionLayer` returns a tensor that has undergone feature fusion. The shape of the output
    tensor is `(B, max_length, features)`, where `features` is a hyperparameter.
    """

    def __init__(self, vocab_size: Dict[str, int], config: SublayersConfig):
        super().__init__()

        self.keys = vocab_size.keys()

        if len(config.fc_out) != 2:
            raise ListLengthError("fc_out", expect=2, got=len(config.fc_out))
        if "latlng" not in config.fc_out or "others" not in config.fc_out:
            raise MissingKeyError("fc_out", expect="latlng, others")

        fc_out_latlng, fc_out_others = config.fc_out["latlng"], config.fc_out["others"]

        if fc_out_latlng <= 0:
            raise NonPositiveIntegerError("fc_out['latlng']", fc_out_latlng)
        if fc_out_others <= 0:
            raise NonPositiveIntegerError("fc_out['others']", fc_out_others)

        sum_fc_out = int(
            torch.tensor(
                [
                    fc_out_latlng if key == "latlng" else fc_out_others
                    for key in self.keys
                    if key != "mask"
                ]
            )
            .sum()
            .item()
        )

        # FC: (sum(fc_out) + latent_dim) -> (features)
        self.feature_fusion_fc_noise = FCLayer(
            in_features=sum_fc_out + config.latent_dim,
            out_features=config.features,
            activation="relu",
        )

        # FC: (sum(fc_out)) -> (features)
        self.feature_fusion_fc = FCLayer(
            in_features=sum_fc_out,
            out_features=config.features,
            activation="relu",
        )

    def forward(
        self, embeddings: List[Tensor], noise: Optional[Tensor] = None
    ) -> Tensor:
        """
        `forward` performs the forward pass of the feature fusion layer. It takes in the input
        embeddings and noise (if provided), concatenates them along the feature dimension, applies
        fully connected layers to fuse the features, and returns the fused features.
        """

        # <- embeddings: [N-1; (B, max_length, fc_out)]
        # <- noise: (B, latent_dim)

        # concated: (B, max_length, sum(fc_out))
        concated: Tensor = torch.cat(embeddings, dim=2)

        # permuted: (max_length, B, sum(fc_out))
        permuted = concated.permute(1, 0, 2)

        # feature_fused: [max_length; (B, features)]
        feature_fused = (
            [self.feature_fusion_fc(x) for x in permuted]  # x: (B, sum(fc_out))
            if noise is None
            else [
                self.feature_fusion_fc_noise(torch.cat([x, noise], dim=-1))
                for x in permuted  # x: (B, sum(fc_out))
            ]
        )

        # stacked: (B, max_length, features)
        stacked = torch.stack(feature_fused, dim=1)

        # -> stacked: (B, max_length, features)
        return stacked


class TransformerModelingLayer(nn.Module):
    """
    # Transformer Modeling Layer

    `TransformerModelingLayer` implements a convolutional sparse transformer neural network for time
    series modeling for both generator and discriminator. It takes in a `SublayersConfig` object
    that specifies various hyperparameters of the model. The output of the `forward` method is a
    tuple containing two tensors: a tensor of shape `(batch_size, max_length, extraction_dim)` that
    represents the embedding of the input time series, and a tensor of shape
    `(batch_size, extraction_dim)` that represents the output of the model.

    ## Parameters

    `TransformerModelingLayer` has one required parameter:

    - `config` (SublayersConfig): an object containing the configuration information for this layer.

    ## Input

    `TransformerModelingLayer` takes in a tensor of shape `(batch_size, max_length, features)`
    representing a batch of input time series, where `max_length` is the maximum length of the time
    series, and `features` is the number of features per time step.

    ## Output

    `TransformerModelingLayer` returns a tuple containing two tensors:

    - `transformer_embedding` (`Tensor`): a tensor of shape
    `(batch_size, max_length, extraction_dim)` representing the embedding of the input time series.
    - `outputs` (`Tensor`): a tensor of shape `(batch_size, extraction_dim)` representing the output
    of the model.
    """

    def __init__(self, config: SublayersConfig):
        super().__init__()

        if config.extraction_dim <= 0:
            raise NonPositiveIntegerError("extraction_dim", config.extraction_dim)
        if config.transformer.kernel_size <= 0:
            raise NonPositiveIntegerError("kernel_size", config.transformer.kernel_size)

        # Embedding: (2) -> (extraction_dim)
        self.input_embedding = ContextEmbedding(
            in_channels=2,
            out_channels=config.extraction_dim,
            kernel_size=config.transformer.kernel_size,
        )

        if config.transformer.num_embeddings <= 0:
            raise NonPositiveIntegerError(
                "num_embeddings", config.transformer.num_embeddings
            )

        # Embedding: (num_embeddings, extraction_dim)
        self.positional_embedding = nn.Embedding(
            config.transformer.num_embeddings, config.extraction_dim
        )

        if config.features <= 0:
            raise NonPositiveIntegerError("features", config.features)
        if config.transformer.num_heads <= 0:
            raise NonPositiveIntegerError("num_heads", config.transformer.num_heads)
        if config.transformer.num_layers <= 0:
            raise NonPositiveIntegerError("num_layers", config.transformer.num_layers)

        # TransformerEncoderLayer: (d_model, num_heads)
        self.transformer_decoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=config.extraction_dim,
                config=config.transformer,
            ),
            num_layers=config.transformer.num_layers,
        )

        # FC: (features) -> (1)
        self.extraction = FCLayer(
            in_features=config.features,
            out_features=1,
            activation="relu",
        )

        # FC: (extraction_dim) -> (extraction_dim)
        self.fc_layer = FCLayer(
            in_features=config.extraction_dim,
            out_features=config.extraction_dim,
            activation="relu",
        )

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        `forward` performs the forward pass of the transformer. It takes in a tensor `inputs` of
        shape `(batch_size, max_length, features)` representing a batch of input time series, and
        returns a tuple of two tensors: `transformer_embedding` and `outputs`. It first generates
        time covariates and attention masks, and concatenates them with the input tensor. Then it
        performs embedding, positional encoding, feature extraction, and TransformerEncoder.
        Finally, it applies two fully connected layers to generate the final output.
        """

        # <- (B, max_length, features)

        # time_covariate: (B, max_length, features)
        time_covariate: Tensor = (
            torch.arange(inputs.shape[1])
            .view(1, -1, 1)
            .repeat(inputs.shape[0], 1, inputs.shape[2])
        )

        # attention_masks: (max_length, max_length)
        attention_masks: Tensor = torch.triu(
            torch.ones(inputs.shape[1], inputs.shape[1]) * float("-inf"), diagonal=1
        )

        # z_inputs: (B, 2, max_length, features)
        z_inputs = torch.cat((inputs.unsqueeze(1), time_covariate.unsqueeze(1)), 1)

        # z_embedding: (B, extraction_dim, max_length, features)
        z_embedding: Tensor = self.input_embedding(z_inputs)

        # z_embedding: (B, max_length, extraction_dim, features)
        z_embedding = z_embedding.permute(0, 2, 1, 3)

        # positional_embeddings: (B, max_length, features, extraction_dim)
        positional_embeddings: Tensor = self.positional_embedding(
            time_covariate.type(torch.long)
        )
        # positional_embeddings: (B, max_length, extraction_dim, features)
        positional_embeddings = positional_embeddings.permute(0, 1, 3, 2)

        # embedding: (B, max_length, extraction_dim, features)
        embedding: Tensor = z_embedding + positional_embeddings

        # embedding:(B, max_length, extraction_dim, 1)
        embedding = self.extraction(embedding)

        # embedding: (B, max_length, extraction_dim)
        embedding = torch.squeeze(embedding, -1)

        # transformer_embedding: (B, max_length, extraction_dim)
        transformer_embedding: Tensor = self.transformer_decoder(
            embedding, attention_masks
        )

        # outputs: (B, extraction_dim)
        outputs = self.fc_layer(transformer_embedding[:, -1, :])

        # -> transformer_embedding: (B, max_length, extraction_dim)
        # -> outputs: (B, extraction_dim)
        return transformer_embedding, outputs


class LSTMModelingLayer(nn.Module):
    """
    # LSTM Modeling Layer

    `LSTMModelingLayer` is a PyTorch module that implements an LSTM layer for modeling sequential
    data for both generator and discriminator. It takes in a configuration object of type
    `SublayersConfig` which specifies the number of input features and the number of output
    dimensions. It is responsible for processing the input feature representations to generate
    hidden state representations that capture the temporal dependencies in the input sequence.

    ## Parameters

    `LSTMModelingLayer` has one required parameter:

    - `config` (SublayersConfig): an object containing the configuration information for this layer.

    ## Input

    `LSTMModelingLayer` takes in a 3D tensor `emb_traj` of shape
    `(batch_size, max_length, features)`, where `max_length` is the length of the sequence, and
    `features` is the number of input features.

    ## Output

    `LSTMModelingLayer` returns a tuple of two tensors:

    - `hidden` (`Tensor`): a 3D tensor of shape `(batch_size, max_length, extraction_dim)`
    representing the hidden states of the LSTM layer.
    - `cell` (`Tensor`): a 2D tensor of shape `(batch_size, extraction_dim)` representing the final
    cell state of the LSTM layer.
    """

    def __init__(self, config: SublayersConfig):
        super().__init__()

        if config.features <= 0:
            raise NonPositiveIntegerError("features", config.features)
        if config.extraction_dim <= 0:
            raise NonPositiveIntegerError("extraction_dim", config.extraction_dim)

        # lstm: (features) -> (extraction_dim)
        self.lstm = nn.LSTM(
            input_size=config.features,
            hidden_size=config.extraction_dim,
            batch_first=True,
        )

    def forward(self, emb_traj: Tensor) -> Tuple[Tensor, Tensor]:
        """
        `forward` performs the forward pass of the LSTM layer on the input tensor `emb_traj`. It
        first applies the LSTM layer to the input tensor and obtains the hidden and cell states.
        Then, it squeezes the cell state to remove the first dimension (which corresponds to the
        number of layers in the LSTM). Finally, it returns the hidden and cell states as a tuple of
        two tensors.
        """

        # <- emb_traj: (B, max_length, features)

        hidden, (cell, _) = self.lstm(emb_traj)
        cell: Tensor = cell.squeeze(dim=0)

        # -> hidden: (B, max_length, extraction_dim)
        # -> cell: (B, extraction_dim)
        return hidden, cell


class RegressionLayer(nn.Module):
    """
    # Regression Layer

    `RegressionLayer` implements a regression layer to predict the output based on the input for
    generator. It takes in a vocabulary size and a `SublayersConfig` object. The output of `forward`
    method is a list of tensors, each tensor representing a different prediction. The output is
    predicted based on the output of the temporal extraction modeling layer, i.e, LSTM modeling
    layer or Transformer modeling layer.

    ## Parameters

    `RegressionLayer` has two parameters:

    - `vocab_size` (Dict[str, int]): a dictionary containing the number of unique elements in each
    category.
    - `config` (SublayersConfig): an object containing the configuration information for this layer.

    ## Input

    `RegressionLayer` takes in two tensors:

    - `temporal` (Tensor): the input tensor of shape `(batch_size, max_length, extraction_dim)`.
    - `mask` (Tensor): the mask tensor of shape `(batch_size, max_length)`.

    ## Output

    `RegressionLayer` returns a list of tensors, each tensor represents a different prediction. The
    length of the list is the same as the number of output keys in `vocab_size` dictionary.
    """

    def __init__(self, vocab_size: Dict[str, int], config: SublayersConfig):
        super().__init__()

        self.keys = vocab_size.keys()

        if config.extraction_dim <= 0:
            raise NonPositiveIntegerError("extraction_dim", config.extraction_dim)
        if config.scale_factor <= 0:
            raise NonPositiveFloatError("scale_factor", config.scale_factor)

        self.scale_factor = config.scale_factor

        self.regression_fcs = nn.ModuleDict(
            {  # FC: (extraction_dim) -> (fc_out)
                key: FCLayer(config.extraction_dim, 2, "tanh")
                if key == "latlng"
                else FCLayer(config.extraction_dim, vocab_size[key], "sigmoid")
                for key in self.keys
                if key != "mask"
            }
        )

    def forward(self, temporal: Tensor, mask: Tensor) -> List[Tensor]:
        """
        `forward` performs the forward pass of the regression layer. It applies fully connected
        layers to the input tensor and generates a list of outputs. For each output, if its key is
        `"latlng"`, its output tensor is multiplied by a scale factor specified in `config`,
        otherwise the tensor is passed through a sigmoid activation function.
        """

        # <- temporal: (B, max_length, extraction_dim)
        # <- mask: (B, max_length)

        outputs: List[Tensor] = []
        for key in self.keys:
            if key == "mask":
                # mask: (B, max_length, 1)
                mask = mask.unsqueeze(-1)
                outputs.append(mask)
            else:
                # output: (B, max_length, fc_out)
                output: Tensor = self.regression_fcs[key](temporal)
                scale_factor = self.scale_factor if key == "latlng" else 1
                outputs.append(output.mul(scale_factor))

        # -> return outputs: [N; (B, max_length, N)]
        return outputs


class ClassificationLayer(nn.Module):
    """
    # Classification Layer

    `ClassificationLayer` implements a fully-connected layer for classification for discriminator.
    It takes in a tensor of shape `(batch_size, max_length, extraction_dim)` and returns a tensor of
    shape `(batch_size, max_length, 1)` where each value is in the range [0, 1], representing the
    probability of a certain class.

    ## Parameters

    `ClassificationLayer` has one required parameter:

    - `config` (SublayersConfig): an object containing the configuration information for this layer.

    ## Input

    `ClassificationLayer` takes in a tensor of shape `(batch_size, max_length, extraction_dim)`
    representing the input features for each time step.

    ## Output

    `ClassificationLayer` returns a tensor of shape `(batch_size, max_length, 1)` representing the
    probability of a certain class at each time step.
    """

    def __init__(self, config: SublayersConfig):
        super().__init__()

        if config.extraction_dim <= 0:
            raise NonPositiveIntegerError("extraction_dim", config.extraction_dim)

        self.classification_fc = FCLayer(config.extraction_dim, 1, "sigmoid")

    def forward(self, temporal: Tensor) -> Tensor:
        """
        `forward` performs the forward pass of the layer. It takes in a temporal tensor of shape
        `(batch_size, max_length, extraction_dim)` and applies a fully connected layer with a
        sigmoid activation function to obtain the classification probabilities. It then returns the
        tensor of shape `(batch_size, max_length, 1)` representing the classification probabilities.
        """

        # <- temporal: (B, max_length, extraction_dim)

        # -> return: (B, max_length, 1)
        return self.classification_fc(temporal)
