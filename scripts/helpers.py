from itertools import groupby
from typing import IO

def fasta_iter(fasta_file : IO):
    """
    Modified from: https://www.biostars.org/p/710/
    """
    faiter = (x[1] for x in groupby(fasta_file, lambda line: line[0] == ">"))
    for header in faiter:
        headerStr = header.__next__()[1:].strip()
        seq = "".join(s.strip() for s in faiter.__next__())
        yield (headerStr, seq)