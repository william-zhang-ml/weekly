"""All sorts of random things."""
import git
import pandas as pd


def get_repo_hash() -> str:
    """
    Returns:
        str: long hash of the repo
    """
    repo = git.Repo(search_parent_directories=True)
    return repo.head.commit.hexsha


def get_hyperparameters(file: str, index: int) -> dict:
    """Get a specific hyperparameter row from a CSV table.

    Args:
        file (str): CSV table of hyperparmeters
        index (int): row to get

    Returns:
        dict: hyperparameter key-value pairs
    """
    hparams = {}
    for key, val in pd.read_csv(file).loc[index].dropna().to_dict().items():
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        hparams[key] = val
    return hparams
