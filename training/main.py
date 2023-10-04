""" Example training script. """
from datetime import datetime
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import yaml


def get_network(arch_name: str) -> nn.Module:
    """Dedicated function for swapping network architectures by name.

    Args:
        arch_name (str): network architecture to instantiate

    Returns:
        nn.Module: neural network instance
    """
    net = getattr(models, arch_name)()
    net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 10)
    return net


def export_network(to_export: nn.Module, path: str) -> None:
    """Export network to ONNX file.

    Args:
        to_export (nn.Module): network to export
        path (str): file path to write to
    """
    torch.onnx.export(
        to_export,
        torch.rand(1, 1, 128, 128),
        path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {
                0: 'batch',
                2: 'row',
                3: 'col'
            }
        }
    )


if __name__ == '__main__':
    # get ready to write new experiment outputs
    TAG = datetime.now().strftime('%Y%m%d-%H%M%S')
    OUTPUT_DIR = Path('./outputs/') / TAG
    os.makedirs(OUTPUT_DIR)
    BOARD = SummaryWriter(log_dir=f'_tensorboard/{TAG}')

    with open('./configs/dev.yaml', 'r', encoding='utf-8') as file:
        CONFIG = yaml.safe_load(file)
    with open(OUTPUT_DIR / 'config.yml', 'w', encoding='utf-8') as file:
        yaml.safe_dump(CONFIG, file)

    # extract required config values
    ARCH_NAME = CONFIG['arch_name']

    network = get_network(ARCH_NAME)

    # training teardown - make analysis products and write ONNX weights
    export_network(network, OUTPUT_DIR / 'final.onnx')
