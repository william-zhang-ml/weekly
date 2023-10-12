""" Implements a utility class for interacting with the output directories. """
from datetime import datetime
import os
from pathlib import Path
import torch


class Output:
    """Utility class for interacting with output directories. """
    def __init__(self, root: str, tag: str = None):
        self.root = Path(root)
        if tag is None:
            self.tag = datetime.now().strftime('%Y%m%d-%H%M%S')
        else:
            self.tag = tag
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        """Highest directory that is run-specific. """
        return self.root / self.tag

    @property
    def checkpoint_dir(self) -> Path:
        """Where training checkpoints should go. """
        return self.output_dir / 'checkpoints'

    def get_latest_checkpoint(self) -> dict:
        """Load the latest checkpoint (curr alphabetically).

        Returns:
            dict: checkpoint variables and state dicts
        """
        checkpoints = sorted(self.checkpoint_dir.glob('*.pt'))
        return torch.load(checkpoints[-1])
