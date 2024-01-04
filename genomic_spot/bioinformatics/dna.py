"""Class to compute info about nucleotide sequences."""

from collections import defaultdict
from itertools import product
from typing import Dict

from ..helpers import count_kmers


class DNA:
    """Calculations on a DNA sequence.

    Typical usage:
    ```
    dna_calc = DNA(dna)
    dna_metrics = DNA.nucleotide_metrics() # all metrics
    kmer_freqs = DNA.count_canonical_kmers(k=1) # k-mer frequency
    ```
    """

    BASE_TYPE = {"C": "Y", "T": "Y", "A": "R", "G": "R"}

    def __init__(self, dna: str):
        """
        Args:
            dna: A DNA sequence, gene or genome
        """
        self.sequence = dna
        self.length = len(self.sequence)
        self._nt_1mer_frequencies = None
        self._nt_2mer_frequencies = None

    def reverse_complement(self, sequence):
        """Returns the reverse complement of a DNA sequence"""
        complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
        return "".join([complement.get(nt, "") for nt in sequence[::-1]])

    def make_canonical_kmers_dict(self, k: int):
        """Creates a dictionary that where a k-mer and its reverse
        complement k-mer are keys, and the k-mer is the value, for
        all k-mers at a given k. E.g.: {'AA' : 'TT', 'AA' : 'AA'}
        """
        canonical_kmers_dict = {}
        kmers_sorted = sorted(["".join(i) for i in product(["A", "T", "C", "G"], repeat=k)])
        for kmer in kmers_sorted:
            if kmer not in canonical_kmers_dict.keys():
                canonical_kmers_dict[kmer] = kmer
                canonical_kmers_dict[self.reverse_complement(kmer)] = kmer
        return canonical_kmers_dict

    def count_canonical_kmers(self, k: int) -> Dict[str, float]:
        """Returns counts for canonical k-mers only, e.g. 'AA' records
        counts for both 'AA' and its reverse complement 'TT'. Canonical
        is determined by lexicographic order.
        """
        kmers_count = count_kmers(self.sequence, k)

        canonical_kmers_dict = self.make_canonical_kmers_dict(k)
        canonical_kmers_count = defaultdict(int)
        for kmer, count in kmers_count.items():
            canonical_kmer = canonical_kmers_dict.get(kmer, None)
            if canonical_kmer is not None:
                canonical_kmers_count[canonical_kmer] += count
        return dict(canonical_kmers_count)

    def nt_1mer_frequencies(self) -> Dict[str, float]:
        """Count frequencies of canonical 1-mers"""
        if self._nt_1mer_frequencies is None:
            kmers_count = self.count_canonical_kmers(k=1)
            n_kmers = sum(kmers_count.values())
            self._nt_1mer_frequencies = {k: float(v / n_kmers) for k, v in kmers_count.items()}
        return self._nt_1mer_frequencies

    def nt_2mer_frequencies(self) -> Dict[str, float]:
        """Count frequencies of canonical 2-mers"""
        if self._nt_2mer_frequencies is None:
            kmers_count = self.count_canonical_kmers(k=2)
            n_kmers = sum(kmers_count.values())
            self._nt_2mer_frequencies = {k: float(v / n_kmers) for k, v in kmers_count.items()}
        return self._nt_2mer_frequencies

    def purine_pyrimidine_transition_freq(self) -> float:
        """Frequency of purine-pyrimidine transitions. An example
        transition is AT, which is purine (R) to pyrimidine (Y).
        """
        transition_frequency = 0
        for sequence, freq in self.nt_2mer_frequencies().items():
            if self.BASE_TYPE[sequence[0]] != self.BASE_TYPE[sequence[1]]:
                transition_frequency += freq

        return float(transition_frequency)

    def nucleotide_metrics(self) -> Dict[str, float]:
        """Computes a dictionary with all metrics for a DNA sequence"""

        sequence_metrics = {
            "nt_length": self.length,
            "pur_pyr_transition_freq": self.purine_pyrimidine_transition_freq(),
        }

        # Must prepend with "nt_" because code overlaps with aas
        for nt, count in self.nt_1mer_frequencies().items():  # 2 variables
            sequence_metrics["nt_{}".format(nt)] = float(count)

        return sequence_metrics
