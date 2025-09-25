# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""Vanilla MLP layer as a drop-in replacement for EquiLinear."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


class VanillaMLP(nn.Module):
    """Vanilla MLP layer that mimics the EquiLinear interface.

    This is a drop-in replacement for EquiLinear that uses standard MLPs instead of
    equivariant linear operations. It flattens the multivector inputs, processes them
    through a standard MLP, and reshapes the outputs back to multivector format.

    WARNING: This layer breaks Pin(3,0,1) equivariance and should only be used when
    you want to experiment with non-equivariant architectures.

    The forward pass maps multivector inputs with shape (..., in_channels, 16) and
    optional scalar inputs with shape (..., in_s_channels) to multivector outputs
    with shape (..., out_channels, 16) and optional scalar outputs.

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels
    out_mv_channels : int
        Output multivector channels
    in_s_channels : int or None
        Input scalar channels. If None, no scalars are expected nor returned.
    out_s_channels : int or None
        Output scalar channels. If None, no scalars are expected nor returned.
    hidden_dim : int
        Hidden dimension for the MLP. If None, uses 2 * total_input_dim.
    num_layers : int
        Number of hidden layers in the MLP.
    activation : str
        Activation function to use. Options: 'gelu', 'relu', 'swish'.
    bias : bool
        Whether to use bias terms in linear layers.
    dropout_prob : float or None
        Dropout probability. If None, no dropout is applied.
    initialization : str
        Initialization scheme to match EquiLinear options.
        Options: "default", "small", "unit_scalar", "almost_unit_scalar"
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "gelu",
        bias: bool = True,
        dropout_prob: Optional[float] = None,
        initialization: str = "default",
    ) -> None:
        super().__init__()

        # Store configuration
        self.in_mv_channels = in_mv_channels
        self.out_mv_channels = out_mv_channels
        self.in_s_channels = in_s_channels or 0
        self.out_s_channels = out_s_channels or 0

        # Calculate dimensions
        self.total_input_dim = in_mv_channels * 16 + self.in_s_channels
        self.total_output_dim = out_mv_channels * 16 + self.out_s_channels

        if hidden_dim is None:
            hidden_dim = max(2 * self.total_input_dim, 64)

        # Build MLP layers
        layers = []

        # Input layer
        layers.append(nn.Linear(self.total_input_dim, hidden_dim, bias=bias))
        layers.append(self._get_activation(activation))

        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(self._get_activation(activation))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_dim, self.total_output_dim, bias=bias))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self.reset_parameters(initialization)

        # Count nominal FLOPs (rough estimate)
        self.nominal_flops_per_token = self._estimate_flops()

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _estimate_flops(self) -> int:
        """Estimate FLOPs per token for this layer."""
        total_params = sum(p.numel() for p in self.mlp.parameters())
        # Rough estimate: 2 FLOPs per parameter (multiply + add)
        return total_params * 2

    def forward(
        self, multivectors: torch.Tensor, scalars: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass through vanilla MLP.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are expected. Otherwise None.
        """
        batch_shape = multivectors.shape[:-2]

        # Flatten multivectors: (..., in_mv_channels, 16) -> (..., in_mv_channels * 16)
        mv_flat = multivectors.reshape(*batch_shape, -1)

        # Concatenate with scalars if present
        if scalars is not None:
            if self.in_s_channels == 0:
                raise ValueError("Received scalar inputs but in_s_channels=0")
            inputs = torch.cat([mv_flat, scalars], dim=-1)
        else:
            if self.in_s_channels > 0:
                # Pad with zeros if scalars expected but not provided
                zeros = torch.zeros(
                    *batch_shape,
                    self.in_s_channels,
                    dtype=multivectors.dtype,
                    device=multivectors.device,
                )
                inputs = torch.cat([mv_flat, zeros], dim=-1)
            else:
                inputs = mv_flat

        # Forward pass through MLP
        outputs = self.mlp(inputs)

        # Split outputs back into multivectors and scalars
        mv_output_size = self.out_mv_channels * 16

        # Reshape multivector outputs
        outputs_mv = outputs[..., :mv_output_size].reshape(*batch_shape, self.out_mv_channels, 16)

        # Extract scalar outputs if expected
        if self.out_s_channels > 0:
            outputs_s = outputs[..., mv_output_size:]
        else:
            outputs_s = None

        return outputs_mv, outputs_s

    def reset_parameters(
        self,
        initialization: str,
        gain: float = 1.0,
        additional_factor: float = 1.0 / np.sqrt(3.0),
    ) -> None:
        """Initialize the weights of the MLP.

        Parameters
        ----------
        initialization : str
            Initialization scheme. Options: "default", "small", "unit_scalar", "almost_unit_scalar"
        gain : float
            Gain factor for initialization.
        additional_factor : float
            Additional scaling factor.
        """
        if initialization not in {"default", "small", "unit_scalar", "almost_unit_scalar"}:
            raise ValueError(f"Unknown initialization scheme {initialization}")

        # Compute scaling factor based on initialization scheme
        if initialization == "default":
            scale_factor = gain * additional_factor
        elif initialization == "small":
            scale_factor = 0.1 * gain * additional_factor
        elif initialization in ["unit_scalar", "almost_unit_scalar"]:
            # For unit_scalar schemes, use smaller scale to encourage sparse outputs
            scale_factor = 0.1 * gain * additional_factor

        # Initialize all linear layers
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot uniform initialization scaled by our factor
                fan_in = module.in_features
                bound = scale_factor * np.sqrt(6.0 / fan_in)
                nn.init.uniform_(module.weight, -bound, bound)

                if module.bias is not None:
                    if initialization in ["unit_scalar", "almost_unit_scalar"]:
                        # For unit_scalar, bias the first component (scalar) toward 1
                        # This is a rough approximation since we don't know which outputs
                        # correspond to scalar components after flattening
                        nn.init.uniform_(module.bias, -0.1, 0.1)
                        # Set first few bias terms slightly positive to encourage scalar output
                        if module.bias.shape[0] >= self.out_mv_channels:
                            with torch.no_grad():
                                # Bias the scalar components (every 16th element starting from 0)
                                for i in range(
                                    0, min(module.bias.shape[0], self.out_mv_channels * 16), 16
                                ):
                                    module.bias[i] += (
                                        1.0 if initialization == "unit_scalar" else 0.5
                                    )
                    else:
                        nn.init.uniform_(module.bias, -bound, bound)

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (
            f"in_mv_channels={self.in_mv_channels}, "
            f"out_mv_channels={self.out_mv_channels}, "
            f"in_s_channels={self.in_s_channels}, "
            f"out_s_channels={self.out_s_channels}, "
            f"total_params={sum(p.numel() for p in self.parameters())}"
        )
