"""General helper functions"""

from collections import Counter
from itertools import groupby
from typing import (
    IO,
    Tuple,
)

import numpy as np


def fasta_iter(fasta_file: IO) -> Tuple[str, str]:
    """Iterable yielding FASTA header and sequence

    Modified from: https://www.biostars.org/p/710/

    Args:
        fasta_file: File object for FASTA file

    Yields:
        headerStr: Header following without >
        seq: FASTA sequence
    """
    faiter = (x[1] for x in groupby(fasta_file, lambda line: line[0] == ">"))
    for header in faiter:
        headerStr = header.__next__()[1:].strip()
        seq = "".join(s.strip() for s in faiter.__next__())
        yield (headerStr, seq)


def count_kmers(sequence: str, k: int) -> dict:
    """Returns counts of every observed k-mer at specific k.

    Args:
        sequence: Sequence, protein or nucleotide
        k: Length of string

    Returns:
        Dictionary of k-mer counts e.g. {'AA' : 2, ...}
    """
    kmers_count = Counter([sequence[i : i + k] for i in range(len(sequence) - k + 1)])
    return dict(kmers_count)
