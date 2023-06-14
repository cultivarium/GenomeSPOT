"""General helper functions"""

from collections import Counter
from itertools import groupby
from typing import IO, Tuple

import numpy as np

def fasta_iter(fasta_file : IO) -> Tuple[str, str]:
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

def count_kmers(sequence : str, k : int) -> dict:
    """Returns counts of every observed k-mer at specific k.
    
    Args:
        sequence: Sequence, protein or nucleotide 
        k: Length of string
    
    Returns:
        Dictionary of k-mer counts e.g. {'AA' : 2, ...}
    """
    kmers_count = Counter([sequence[i:i+k] for i in range(len(sequence) - k + 1)])
    return dict(kmers_count)

def nonnull(list_ : list):
    """Returns array without NaN values"""
    X = np.array(list_)
    return X[~np.isnan(X)]

def bin_midpoints(data : np.array, bins: np.array):
    """Returns counts per bin keyed by bin midpoint"""
    bin_counts = []
    bin_midpoints = []
    counts = Counter(np.digitize(data, bins))
    for bin_idx in range(len(bins) - 1):
        bin_counts.append(counts.get(bin_idx, 0))
        bin_midpoints.append(round(np.mean([bins[bin_idx], bins[bin_idx + 1]]), 3))
    return dict(zip(bin_midpoints, bin_counts))

def onehot_range(arr, min_bin : float, max_bin : float, step : float) -> dict:
    """Returns a dictionary of presence or absence in a bin"""
    onehot_dict = {}
    for bin_floor in np.arange(min_bin, max_bin, step):
        if min(arr, default=-1000) <= bin_floor <= max(arr, default=-1000):
            onehot_dict[str(bin_floor)] = 1
        else:
            onehot_dict[str(bin_floor)] = 0
    return onehot_dict


def gtdb_taxonomy_to_dict(taxstring : str) -> dict:
    """Convert GTDB taxstring to a dictionary.
    
    Args:
        taxstring: A GTDB taxstring in the following format: 
            d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;...
    Returns:
        taxonomy_dict: A dictionary is keyed by the following ranks: domain, phylum, class, 
            order, family, genus, and species.
    """
    GTDB_ABBREV = {'d' : 'domain', 'p' : 'phylum', 'c' : 'class', 'o' : 'order', 'f' : 'family' , 'g' : 'genus', 's' : 'species'}
    taxonomy_dict = {}    
    for level in taxstring.strip().split(';'):
        abbrev, taxon = level.split('__')
        taxonomy_dict[GTDB_ABBREV[abbrev]] = taxon
    return taxonomy_dict

def gtdb_accession_to_ncbi(accession : str, 
                           make_genbank : bool = True, 
                           remove_version : bool = True) -> str:
    """Convert GTDB 'accession' into NCBI accession.

    Options allow different formats.

    Args:
        accession: GTDB accession e.g. RS_GCF_016456235.1
        make_genbank: Replace the initial 'GCF_' with 'GCA_'
        remove_version: Remove the terminal '.#'
    Returns:
        ncbi_accession : NCBI accession e.g. GCA_016456235
    """

    ncbi_accession = accession[3:]
    if make_genbank:
        ncbi_accession = ncbi_accession.replace('GCF_', 'GCA_')
    if remove_version:
        ncbi_accession = ncbi_accession[:-2]
    return ncbi_accession
    