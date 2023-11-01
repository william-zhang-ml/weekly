"""All sorts of random things."""
import git


def get_hash() -> str:
    """
    Returns:
        str: long hash of the repo
    """
    repo = git.Repo(search_parent_directories=True)
    return repo.head.commit.hexsha
