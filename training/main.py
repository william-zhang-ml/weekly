""" Example training script. """
from datetime import datetime
import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
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
    BATCH_SIZE = CONFIG['batch_size']
    EPOCH_PER_CYCLE = CONFIG['epoch_per_cycle']
    LABEL_SMOOTHING = CONFIG['label_smoothing']
    MNIST_PATH = CONFIG['mnist_path']
    NUM_EPOCH = CONFIG['num_epoch']

    # initialize training variables
    dataset = MNIST(MNIST_PATH, transform=transforms.ToTensor())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    criteria = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    network = get_network(ARCH_NAME)
    optimizer = optim.AdamW(network.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, EPOCH_PER_CYCLE)
    curr_epoch, step = 0, 0

    # train
    epochbar = \
        tqdm(range(curr_epoch, NUM_EPOCH), initial=curr_epoch, total=NUM_EPOCH)
    for i_epoch in epochbar:
        # main training loop
        batchbar = tqdm(loader, leave=True)
        for imgs, labels in batchbar:
            logits = network(imgs)
            pred_labels = logits.argmax(dim=1)
            loss = criteria(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            BOARD.add_scalar('Loss/train', loss.item(), step)
            step += 1
            batchbar.set_postfix({
                'loss': f'{loss:.03f}',
                'acc': f'{100 * (pred_labels == labels).float().mean():.01f}'
            })
            batchbar.close()
            break
        scheduler.step()

    # training teardown - make analysis products and write ONNX weights
    export_network(network, OUTPUT_DIR / 'final.onnx')
    print('DONE!')