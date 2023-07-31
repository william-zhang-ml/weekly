""" Markdown utilities. """
from typing import List


def _write_row(elems: List[str], col_width: List[int]) -> str:
    """Write a '|'-separated markdown row.

    Args:
        elems (List[str]): elements in the row
        col_width (List[str]): width of each column

    Return:
        str: table row as a single markdown string
    """
    padded_elems = [
        curr.ljust(width, ' ')
        for curr, width in zip(elems, col_width)
    ]
    return ''.join(['| ', ' | '.join(padded_elems), ' |'])


def table_to_markdown(table: List[List[str]]) -> str:
    """Write a '|'-separated markdown table.

    Args:
        table (List[List[str]]): table to write

    Returns:
        str: table as a single markdown string
    """
    # compute needed width of each column
    col_width: List[int] = [
        max([len(elem) for elem in col]) for col in zip(*table)
    ]

    # start by writing header row
    markdown = [_write_row(table[0], col_width)]

    # write separation row
    markdown.append(
        _write_row(['-' * width for width in col_width], col_width)
    )

    # write data rows
    for row in table[1:]:
        markdown.append(_write_row(row, col_width))

    return '\n'.join(markdown)
