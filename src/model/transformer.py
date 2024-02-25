"""
## Information

- File Name: transformer.py
- Author: Selene
- Date of Creation: 2023.04.25
- Date of Last Modification: 2023.05.17 (TODO: Update this)
- Python Version: 3.9.13
- License: GNU GPL v3.0
"""

import math
from dataclasses import dataclass

from typing import Tuple
from utils.utils import Lambda, Utils
import torch
import torch.nn.functional as F
from entmax import entmax15, sparsemax
from torch import Tensor, nn


@dataclass
class FeedforwardConfig:
    "Feedforward Config"

    feedforward_dim: int
    dropout: float


@dataclass
class AttentionConfig:
    "Attention Config"

    attention_type: str
    dropout: float


@dataclass
class TransformerConfig:
    "Transformer Config"

    kernel_size: int
    num_heads: int
    num_layers: int
    num_embeddings: int
    dropout: float
    feedforward: FeedforwardConfig
    attention: AttentionConfig


@dataclass
class QKVM:
    "Query, Key, Value, and Mask"

    query: Tensor
    key: Tensor
    value: Tensor
    mask: Tensor


class CausalConv2d(nn.Conv2d):
    """
    # Causal Conv2d Layer

    `CausalConv2d` is a PyTorch `nn.Module` subclass that implements a 2D causal convolution. The
    convolution is "causal" in the sense that the output at each time step only depends on previous
    time steps, but not future time steps.

    ## Parameters

    `CausalConv2d` has three parameters:

    - `in_channels` (int): number of input channels
    - `out_channels` (int): number of output channels
    - `kernel_size` (int): size of the convolution kernel

    ## Input

    `CausalConv2d` takes in a 4D tensor of shape `(batch_size, in_channels, height, width)`.

    ## Output

    `CausalConv2d` returns a 4D tensor of shape `(batch_size, out_channels, height, width)`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(in_channels, out_channels, kernel_size)

        self._padding = kernel_size - 1

    def forward(self, inputs: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        """
        `forward` performs the forward pass of the 2D causal convolution. Given an input tensor
        `inputs`, it applies zero padding to the left and top of the tensor, with the amount of
        padding equal to `kernel_size - 1`. Then, it calls the `forward` method of the parent
        `nn.Conv2d` class, passing in the padded tensor. The output of the convolution is returned.
        """

        return super().forward(F.pad(inputs, (self._padding, 0, self._padding, 0)))


class ContextEmbedding(nn.Module):
    """
    # Context Embedding

    `ContextEmbedding` is a PyTorch `nn.Module` subclass that implements a context embedding layer.
    The layer takes in a sequence of input vectors and applies a 2D causal convolution to capture
    contextual information.

    ## Parameters

    `ContextEmbedding` has three parameters:

    - `in_channels` (int): number of input channels
    - `out_channels` (int): number of output channels
    - `kernel_size` (int): size of the convolution kernel

    ## Input

    `ContextEmbedding` takes in a 3D tensor of shape `(batch_size, max_length, feature)`, where
    `max_length` is the maximum sequence length and `feature` is the dimensionality of each input
    vector.

    ## Output

    `ContextEmbedding` returns a 3D tensor of shape `(batch_size, max_length, out_channels)`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.model = nn.Sequential(
            CausalConv2d(in_channels, out_channels, kernel_size),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor):
        """
        `forward` performs the forward pass of the context embedding layer. Given an input tensor
        `inputs`, it passes the tensor through a `nn.Sequential` module consisting of a
        `CausalConv2d` layer followed by a `Tanh` activation function. The output of the module is
        returned.
        """

        # inputs: (batch_size, max_length, feature)
        return self.model(inputs)


class LayerNorm(nn.Module):
    """
    # Layer Norm

    `LayerNorm` is a PyTorch `nn.Module` subclass that implements a layer normalization layer.

    ## Parameters

    `LayerNorm` has two parameters:

    - `features` (int): number of input features
    - `eps` (float, optional): a value added to the denominator for numerical stability. Default:
    `1e-6`.

    ## Input

    `LayerNorm` takes in a tensor of shape `(batch_size, ..., features)`, where `...` represents any
    number of additional dimensions.

    ## Output

    `LayerNorm` returns a tensor of the same shape as the input.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, inputs: Tensor):
        """
        `forward` performs the forward pass of the layer normalization layer. Given an input tensor
        `inputs`, it first computes the mean and standard deviation along the last dimension. It
        then applies layer normalization to the input tensor using the mean, standard deviation, and
        learnable scaling and bias parameters. The output of the layer normalization layer is
        returned.
        """

        mu = inputs.mean(-1, keepdim=True)  # pylint: disable=invalid-name
        sigma = inputs.std(-1, keepdim=True).add(self.eps)
        return torch.sub(inputs, mu).div(sigma).mul(self.a_2).add(self.b_2)


class SublayerConnection(nn.Module):
    """
    # Sublayer Connection

    `SublayerConnection` is a PyTorch `nn.Module` subclass that implements a sublayer connection
    layer.

    ## Parameters

    `SublayerConnection` has two parameters:

    - `size` (int): number of input features
    - `dropout` (float): probability of an element to be zeroed. Default: `0.1`

    ## Input

    `SublayerConnection` takes in a tensor of shape `(batch_size, ..., size)`, where `...`
    represents any number of additional dimensions, and a sublayer as an `nn.Module`.

    ## Output

    `SublayerConnection` returns a tensor of the same shape as the input.
    """

    def __init__(self, size: int, dropout: float):
        super().__init__()

        self.norm = LayerNorm(features=size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: Tensor, sublayer: nn.Module):
        """
        `forward` performs the forward pass of the sublayer connection layer. Given an input tensor
        `inputs` and a sublayer `sublayer`, it first normalizes the input tensor using a `LayerNorm`
        layer. The normalized tensor is then passed through the sublayer with dropout applied.
        The output of the sublayer is added to the input tensor using element-wise addition, and
        the result is returned.
        """

        model = nn.Sequential(self.norm, sublayer, self.dropout)

        return inputs.add(model(inputs))


class SparseMHA(nn.Module):
    """
    # Sparse MHA

    `SparseMHA` implements a sparse version of Multi-Head Attention (MHA). It takes in a `QKVM`
    object and returns the attended output.

    ## Parameters

    `SparseMHA` has four parameters:

    - `embed_dim` (int): The input feature dimensionality. Must be divisible by `num_heads`.
    - `num_heads` (int): The number of attention heads.
    - `attn_type` (str): The attention type to use, which can be one of `"softmax"`, `"sparsemax"`,
    or `"entmax15"`.
    - `dropout` (float): The dropout probability.

    ## Input

    `SparseMHA` takes in a `QKVM` object `qkvm`:

    - `qkvm.query` (Tensor): The query tensor of shape `(batch_size, seq_len, embed_dim)`.
    - `qkvm.key` (Tensor): The key tensor of shape `(batch_size, seq_len, embed_dim)`.
    - `qkvm.value` (Tensor): The value tensor of shape `(batch_size, seq_len, embed_dim)`.
    - `qkvm.mask` (Tensor): The mask tensor of shape `(batch_size, seq_len)` indicating which tokens
    are valid (1) and which are not (0).

    ## Output

    `SparseMHA` returns the attended output tensor of shape `(batch_size, seq_len, embed_dim)`.
    """

    def __init__(self, embed_dim: int, num_heads: int, attn_type: str, dropout: float):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.linears = Utils.deepcopy(nn.Linear(embed_dim, embed_dim), 4)
        self.attn_type = attn_type
        self.dropout = nn.Dropout(p=dropout)
        self.scores = None
        self.attn = None

    def forward(self, qkvm: QKVM):
        """
        `forward` first does all the linear projections on `qkvm.query`, `qkvm.key`, and
        `qkvm.value`, using the `nn.Linear` module. Then, the projected tensors are split into
        `num_heads` smaller tensors along the last dimension. The `attention` method is applied on
        each group of smaller tensors using the `attn_type` specified. The weighted sum of the
        `value` tensors is calculated and the resulting tensor is concatenated back into a single
        tensor with shape `(batch_size, seq_len, embed_dim)`. The concatenated tensor is then passed
        through the final linear projection layer and returned.
        """

        mask = qkvm.mask.unsqueeze(0)
        batch_size = qkvm.query.size(0)
        # Do all the linear projections in batch from d_model
        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            for l, x in zip(self.linears, (qkvm.query, qkvm.key, qkvm.value))
        ]

        # Apply attention on all the projected vectors in batch
        attended, self.scores, self.attn = SparseMHA._attention(
            QKVM(query, key, value, mask), self.attn_type, self.dropout
        )
        # Concat using a view and apply a final linear.
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        return self.linears[-1](attended)

    @staticmethod
    def _attention(qkvm: QKVM, attn_type: str, dropout: nn.Dropout):
        """
        `_attention` is a static method that performs the attention calculation. It takes a `QKVM`
        object and the `attn_type` specified and returns the attended output, scores, and attention
        weights.
        """

        d_k = qkvm.query.size(-1)
        scores = torch.matmul(qkvm.query, qkvm.key.transpose(-2, -1)).div(
            math.sqrt(d_k)
        )
        scores = scores.masked_fill(qkvm.mask == 0, -1e9)  # -math.inf is broken

        if attn_type == "softmax":
            p_attn = F.softmax(scores, dim=-1)
        elif attn_type == "sparsemax":
            p_attn = sparsemax(scores, dim=-1)
        elif attn_type == "entmax15":
            p_attn = entmax15(scores, dim=-1)

        weights: Tensor = dropout(p_attn)  # type: ignore
        weights = weights.to(torch.float32)

        return torch.matmul(weights, qkvm.value), scores, weights


class PositionwiseFeedForward(nn.Module):
    """
    # Positionwise Feed Forward

    `PositionwiseFeedForward` implements a feedforward neural network that applies the same linear
    transformation and activation function to each position in a sequence of input embeddings. It
    takes in the dimensionality of the input embeddings, the dimensionality of the feedforward
    layer, and a dropout rate. The output of the feedforward network is a tensor with the same shape
    as the input tensor.

    ## Parameters

    `PositionwiseFeedForward` has three parameters:

    - `d_model` (int): The dimensionality of the input embeddings.
    - `d_ff` (int): The dimensionality of the feedforward layer.
    - `dropout` (float): The dropout rate applied to the output of the feedforward layer.

    ## Input

    `PositionwiseFeedForward` takes in a tensor of shape `(batch_size, seq_len, d_model)`.

    ## Output

    `PositionwiseFeedForward` returns a tensor of shape `(batch_size, seq_len, d_model)`.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, inputs: Tensor):
        """
        `forward` performs the forward pass of the feedforward network. It takes in an input tensor
        `inputs` of shape `(batch_size, seq_len, d_model)` and applies the linear transformation,
        activation function, and dropout to each position in the sequence. It returns a tensor of
        shape `(batch_size, seq_len, d_model)`.
        """

        return self.model(inputs)


class TransformerEncoderLayer(nn.Module):
    """
    # Transformer Encoder Layer

    `TransformerEncoderLayer` implements a single encoder layer in a transformer model. It takes in
    a `d_model` parameter that specifies the feature dimension of the input tensor and a `config`
    parameter that is a `TransformerConfig` instance. The output of the `forward` method is the
    output tensor of the encoder layer.

    ## Parameters

    `TransformerEncoderLayer` has two parameters:

    - `d_model` (int): the feature dimension of the input tensor.
    - `config` (TransformerConfig): a configuration object that stores hyperparameters and
    configurations of the transformer model.

    ## Input

    `TransformerEncoderLayer` takes in two inputs:

    - `inputs` (Tensor): the input tensor to the encoder layer. It has shape
    `(batch_size, max_length, feature)`.
    - `mask` (Tensor): the mask tensor for the input. It has the same shape as `inputs` and is used
    for masking out certain elements of the input tensor during the self-attention computation.

    ## Output

    `TransformerEncoderLayer` returns a tensor that has the same shape as the input tensor `inputs`.
    """

    def __init__(self, d_model: int, config: TransformerConfig):
        super().__init__()

        self.self_attn = SparseMHA(
            embed_dim=d_model,
            num_heads=config.num_heads,
            attn_type=config.attention.attention_type,
            dropout=config.attention.dropout,
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=config.feedforward.feedforward_dim,
            dropout=config.feedforward.dropout,
        )

        self.sublayers = Utils.deepcopy(
            module=SublayerConnection(size=d_model, dropout=config.dropout),
            n_times=2,
        )

        self.size = d_model

    def forward(self, tupled: Tuple[Tensor, Tensor]):
        """
        `forward` performs the forward pass of the encoder layer. It takes in `inputs` and `mask`
        and returns the output tensor after passing through the self-attention and feed-forward
        layers. During the forward pass, the input tensor first passes through a sublayer that
        performs a multi-head self-attention operation, with the output of the attention layer being
        fed to another sublayer that performs a position-wise feed-forward operation. Both sublayers
        have residual connections around them, and the output of the feed-forward layer is passed
        through another residual connection before being returned. The mask tensor is used to mask
        out certain elements of the input tensor during the self-attention computation.
        """

        # just for using nn.Sequential...
        inputs, mask = tupled

        # inputs: (batch_size, max_length, feature)
        inputs = self.sublayers[0](
            inputs, Lambda(lambda x: self.self_attn(QKVM(x, x, x, mask)))
        )
        return self.sublayers[1](inputs, self.feed_forward), mask


class TransformerEncoder(nn.Module):
    """
    # Transformer Encoder

    `TransformerEncoder` implements a Transformer encoder. It takes in `encoder_layer` which is an
    instance of `TransformerEncoderLayer` and `num_layers` which is an integer. The output of
    `TransformerEncoder` is the encoded input.

    ## Parameters

    `TransformerEncoder` has two parameters:

    - `encoder_layer` (TransformerEncoderLayer): an instance of `TransformerEncoderLayer`.
    - `num_layers` (int): an integer representing the number of encoder layers.

    ## Input

    `TransformerEncoder` takes in two arguments:

    - `inputs` (Tensor): the input tensor to be encoded. It has shape
    `(batch_size, max_length, feature)`.
    - `src_mask` (Tensor): the mask tensor for the input. It has shape
    `(batch_size, 1, max_length)`.

    ## Output

    `TransformerEncoder` returns the encoded input tensor that has the same shape as the input
    tensor `inputs`.
    """

    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()

        self.model = nn.Sequential(
            # N stacked encoder layers
            *Utils.deepcopy(module=encoder_layer, n_times=num_layers),
            # get inputs from tuple = (inputs, mask)
            Lambda(lambda tupled: tupled[0]),
            # normalize the output of the final encoder layer
            LayerNorm(features=encoder_layer.size),
        )

    def forward(self, inputs: Tensor, src_mask: Tensor):
        """
        `forward` performs the following steps: For each `layer` in `layers`, `inputs` is passed
        through `layer` along with the `src_mask`. Then the output of the final layer is normalized
        by `LayerNorm`. Finally `encoder_output` is returned.
        """

        return self.model((inputs, src_mask))
