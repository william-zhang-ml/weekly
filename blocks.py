""" Neural network blocks. """
from typing import Dict, List
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
            [curr.numel() for curr in self.parameters() if curr.requires_grad]
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
    """General purpose block of layers with where some args are inferred. """
    def __init__(self,
                 layers: List[str],
                 in_channels: int) -> None:
        super().__init__()
        for i_layer, layer_str in enumerate(layers):
            if layer_str == 'BN':
                self.add_module(str(i_layer), nn.BatchNorm2d(in_channels))
