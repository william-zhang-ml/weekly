""" Example training script. """
import os
import sys
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import yaml
from outputs import Output


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


def export_confmat_fig(net: nn.Module, data: DataLoader, path: str) -> None:
    """Export single-batch confusion matrix graphic.

    Args:
        net (nn.Module): network to infer with
        data (DataLoader): batch generator
        path (str): file path to write to
    """
    imgs, truth = next(iter(data))
    pred = net(imgs).argmax(dim=1)
    confmat = confusion_matrix(truth.cpu().numpy(), pred.cpu().numpy())
    fig, _ = plot_confusion_matrix(confmat, figsize=(8, 8))
    fig.savefig(path)


if __name__ == '__main__':
    # set up a new experiment or continue from existing checkpoint
    try:
        if os.path.isfile(sys.argv[1]):
            # new experiment
            OUTPUT = Output('outputs')
            CHECKPOINT = None
            with open(sys.argv[1], 'r', encoding='utf-8') as file:
                CONFIG = yaml.safe_load(file)
            with open(OUTPUT.config_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(CONFIG, file)
            BOARD = SummaryWriter(log_dir=f'_tensorboard/{OUTPUT.tag}')
        elif os.path.isdir(sys.argv[1]):
            # contimue checkpoint
            root, tag = sys.argv[1].split('/')
            OUTPUT = Output(root, tag)
            CHECKPOINT = OUTPUT.get_checkpoint()
            with open(OUTPUT.config_path, 'r', encoding='utf-8') as file:
                CONFIG = yaml.safe_load(file)
            BOARD = SummaryWriter(log_dir=f'_tensorboard/{OUTPUT.tag}')
        else:
            raise FileNotFoundError()
    except IndexError as exc:
        raise IndexError('Provide config file or output dir.') from exc
    except FileNotFoundError as exc:
        raise FileNotFoundError() from exc

    # extract required config values
    ARCH_NAME = CONFIG['arch_name']
    BATCH_SIZE = CONFIG['batch_size']
    DEVICE = CONFIG['device']
    EPOCH_PER_CYCLE = CONFIG['epoch_per_cycle']
    LABEL_SMOOTHING = CONFIG['label_smoothing']
    MNIST_PATH = CONFIG['mnist_path']
    NUM_EPOCH = CONFIG['num_epoch']

    # initialize training variables
    dataset = MNIST(MNIST_PATH, transform=transforms.ToTensor())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda items: tuple(
            elem.to(DEVICE) for elem in default_collate(items)
        ),
        shuffle=False
    )
    criteria = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    network = get_network(ARCH_NAME).to(DEVICE)
    optimizer = optim.AdamW(network.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, EPOCH_PER_CYCLE)
    curr_epoch, step = 0, 0

    # load checkpoint states into training variables
    if CHECKPOINT is not None:
        network.load_state_dict(CHECKPOINT['network'])
        optimizer.load_state_dict(CHECKPOINT['optimizer'])
        scheduler.load_state_dict(CHECKPOINT['scheduler'])
        curr_epoch = CHECKPOINT['epoch']
        step = CHECKPOINT['step']

    # train
    epochbar = \
        tqdm(range(curr_epoch, NUM_EPOCH), initial=curr_epoch, total=NUM_EPOCH)
    for i_epoch in epochbar:
        # main training loop
        batchbar = tqdm(loader, leave=False)
        for images, labels in batchbar:
            logits = network(images)
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
        scheduler.step()

        # save checkpoint
        torch.save(
            {
                'epoch': i_epoch + 1,
                'step': step,
                'network': network.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            OUTPUT.checkpoint_dir / f'{i_epoch + 1:03d}.pt'
        )

    # training teardown - make analysis products and write ONNX weights
    export_confmat_fig(network, loader, OUTPUT.output_dir / 'confusion-matrix.jpg')
    export_network(network.cpu(), OUTPUT.output_dir / 'final.onnx')
    print('DONE!')
