""" Neural network blocks. """
from typing import Dict, List, Tuple, Union
import torch
from torch import nn


class ModuleMixin:
    """Mix-in for PyTorch nn.Module class that implements nice-to-haves. """
    def count_num_params(self) -> int:
        """
        Returns:
            int: number of trainable parameters
        """
        return sum(
            curr.numel() for curr in self.parameters() if curr.requires_grad
        )

    def get_param_shapes(self) -> Dict[str, torch.Size]:
        """
        Returns:
            List[torch.Size]: shape of each group of named trainable parameters
        """
        shapes = {}
        for param_name, param in self.named_parameters():
            shapes[param_name] = param.shape
        return shapes


class Block(nn.Sequential, ModuleMixin):
    """General purpose block of layers where some args are inferred. """
    def __init__(self,
                 layers: List[str],
                 in_channels: int,
                 out_channels: int = None,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 'same',
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 dropout: float = 0.1,) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        for i_layer, layer_str in enumerate(layers):
            if layer_str == 'BN':
                self.add_module(str(i_layer), nn.BatchNorm2d(in_channels))
            elif layer_str == 'C':
                self.add_module(
                    str(i_layer),
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias
                    ))
                in_channels = out_channels
            elif layer_str == 'D':
                self.add_module(str(i_layer), nn.Dropout(dropout))
            elif layer_str == 'R':
                self.add_module(str(i_layer), nn.ReLU())


class AdaptiveHybridPool2d(nn.Module):
    """Pooling layer that computes and combines adaptive avg & max pooling. """
    def __init__(self,
                 output_size: Union[int, Tuple[int, int]],
                 cat: bool = False) -> None:
        """Initialize pooling branches.

        Args:
            output_size (Union[int, Tuple[int, int]]): target output H/W
            cat (bool): whether to concatenate or average branch outputs
        """
        super().__init__()
        self.cat = cat
        self.branches = nn.ModuleDict({
            'avg': nn.AdaptiveAvgPool2d(output_size),
            'max': nn.AdaptiveMaxPool2d(output_size)
        })

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute and combine adaptive avg & max pooling.

        Args:
            inp (torch.Tensor): feature map to pool

        Returns:
            torch.Tensor: result of pooling
        """
        branch_outp = [branch(inp) for branch in self.branches.values()]
        if self.cat:
            outp = torch.cat(branch_outp, dim=1)
        else:
            outp = 0.5 * (branch_outp[0] + branch_outp[1])
        return outp
