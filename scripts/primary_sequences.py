#!/usr/bin/env python3

"""Classes to compute info about primary sequences.

One class is for proteins, the other for nucleotides. Information about
primary sequences refers to what can be learned from a sequence alone
independent of the gene's function or secondary or tertiary structure.
Examples include nucleotide frequency and the proportion of residues
that are acidic.

Typical usage:

    protein_calc = Protein(protein_sequence)
    protein_metrics = protein_calc.protein_metrics() # all metrics
    pi = protein_calc.isoelectric_point() # individual metric

    dna_calc = DNA(dna)
    dna_metrics = DNA.nucleotide_metrics() # all metrics
    kmer_freqs = DNA.count_canonical_kmers(k=1) # k-mer frequency
"""

from collections import defaultdict
from itertools import product

from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
import numpy as np

from helpers import count_kmers
from signal_peptide import SignalPeptideHMM


class Protein:
    """Calculations on a protein sequence.

    When counting amino acid frequencies, the initial
    Met amino acid (assumed to be present) is removed
    so that changes in protein length do not affect amino
    acid frequencies - e.g. shorter proteins incease Met
    frequency.

    Typical usage:
    ```
    protein_metrics = Protein(protein_sequence).protein_metrics()
    ```
    """

    STANDARD_AMINO_ACIDS = {
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    }

    # citation:
    NH2O_RQEC = {
        "A": 0.369,
        "C": -0.025,
        "D": -0.122,
        "E": -0.107,
        "F": -2.568,
        "G": 0.478,
        "H": -1.825,
        "I": 0.660,
        "K": 0.763,
        "L": 0.660,
        "M": 0.046,
        "N": -0.122,
        "P": -0.354,
        "Q": -0.107,
        "R": 0.072,
        "S": 0.575,
        "T": 0.569,
        "V": 0.522,
        "W": -4.087,
        "Y": -2.499,
    }

    # citation:
    WEIGHTED_ZC = {
        "A": 0,
        "C": 2.0,
        "D": 4,
        "E": 2.0,
        "F": -4.0,
        "G": 2,
        "H": 4.0,
        "I": -6,
        "K": 4.0,
        "L": -6,
        "M": -1.6,
        "N": 4,
        "P": -2.0,
        "Q": 2.0,
        "R": 2.0,
        "S": 1.98,
        "T": 0,
        "V": -4.0,
        "W": -2.0,
        "Y": -2.0,
    }

    # Kyte & Doolittle 1982
    HYDROPHOBICITY = {
        "A": 1.8,
        "R": -4.5,
        "N": -3.5,
        "D": -3.5,
        "C": 2.5,
        "Q": -3.5,
        "E": -3.5,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "L": 3.8,
        "K": -3.9,
        "M": 1.9,
        "F": 2.8,
        "P": -1.6,
        "S": -0.8,
        "T": -0.7,
        "W": -0.9,
        "Y": -1.3,
        "V": 4.2,
    }

    # Membrane protein GRAVY is usually >+0.5 the average GRAVY
    # Kyte & Doolittle 1982
    DIFF_HYDROPHOBICITY_MEMBRANE = 0.5

    THERMOSTABLE_RESIDUES = {"I", "V", "Y", "W", "R", "E", "L"}

    def __init__(self, protein_sequence: str, remove_signal_peptide: True, signal_peptide_model: SignalPeptideHMM):
        """
        Args:
            protein_sequence: Amino acid sequence of one protein
        """
        self.sequence = self._format_protein_sequence(protein_sequence)
        self.length = len(self.sequence)
        self.start_pos = 1  # remove n-terminal Met
        self._aa_1mer_frequencies = None
        self._aa_2mer_frequencies = None
        self.signal_peptide_model = signal_peptide_model
        self.remove_signal_peptide = remove_signal_peptide

    def _format_protein_sequence(self, protein_sequence: str) -> str:
        """Returns a formatted amino acid sequence"""
        return "".join([aa for aa in protein_sequence.strip().upper() if aa in self.STANDARD_AMINO_ACIDS])

    def aa_1mer_frequencies(self) -> dict:
        """Returns count of every amino acid ignoring start methionine"""
        if self._aa_1mer_frequencies is None:
            if self.length > 1:
                self._aa_1mer_frequencies = {
                    k: float(v / len(self.sequence[self.start_pos :]))
                    for k, v in count_kmers(self.sequence[self.start_pos :], k=1).items()
                }
            else:
                self._aa_1mer_frequencies = {}
        return self._aa_1mer_frequencies

    def aa_2mer_frequencies(self) -> dict:
        """Returns count of every amino acid ignoring start methionine"""
        if self._aa_2mer_frequencies is None:
            if self.length > 1:
                self._aa_2mer_frequencies = {
                    k: float(v / len(self.sequence[self.start_pos :]))
                    for k, v in count_kmers(self.sequence[self.start_pos :], k=2).items()
                }
            else:
                self._aa_2mer_frequencies = {}
        return self._aa_2mer_frequencies

    def pi(self) -> float:
        """Compute the isoelectric point (pI) of the protein"""
        if self.length > 0:
            # to-do: remove unnecessary Biopython dependency
            return IsoelectricPoint(self.sequence[self.start_pos :]).pi()
        else:
            return np.nan

    def gravy(self):
        """Compute the hydrophobicity as the
        Grand Average of Hydropathy (GRAVY)
        """
        if self.length > 0:
            return np.mean([self.HYDROPHOBICITY[aa] for aa in self.sequence[self.start_pos :]])
        else:
            return np.nan

    def zc(self) -> float:
        """Computes average carbon oxidation state (Zc) of a
        protein based on a dictionary of amino acids.
        """
        return sum([self.WEIGHTED_ZC[s] for s in self.sequence[self.start_pos :]]) / self.length

    def nh2o(self) -> float:
        """Computes stoichiometric hydration state (nH2O) of a
        protein based on a dictionary of amino acids.
        """
        return sum([self.NH2O_RQEC[s] for s in self.sequence[self.start_pos :]]) / self.length

    def thermostable_freq(self) -> float:
        """Thermostable residues reported by:
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030005
        """
        if self.length > 0:
            return sum([v for k, v in self.aa_1mer_frequencies().items() if k in self.THERMOSTABLE_RESIDUES])
        else:
            return np.nan

    def protein_metrics(self) -> dict:
        """Computes a dictionary with all metrics for a protein"""

        is_exported, signal_end_index = self.signal_peptide_model.predict_signal_peptide(self.sequence)
        if self.remove_signal_peptide is True:
            self.start_pos = signal_end_index + 1
        self.length = len(self.sequence[self.start_pos :])

        sequence_metrics = {
            "pi": self.pi(),
            "zc": self.zc(),
            "nh2o": self.nh2o(),
            "gravy": self.gravy(),
            "thermostable_freq": self.thermostable_freq(),
            "length": self.length,
            "is_exported": is_exported,
        }

        # Must prepend with "aa_" because code overlaps with nts
        for aa, count in self.aa_1mer_frequencies().items():  # 20 variables
            sequence_metrics["aa_{}".format(aa)] = count
        for aa, count in self.aa_2mer_frequencies().items():  # 400 variables
            sequence_metrics["aa_{}".format(aa)] = count

        return sequence_metrics


class DNA:
    """Calculations on a DNA sequence.

    Typical usage:
    ```
    dna_metrics = DNA(dna).nucleotide_metrics()
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

    def count_canonical_kmers(self, k: int) -> dict:
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

    def nt_1mer_frequencies(self) -> dict:
        """Count frequencies of canonical 1-mers"""
        if self._nt_1mer_frequencies is None:
            kmers_count = self.count_canonical_kmers(k=1)
            n_kmers = sum(kmers_count.values())
            self._nt_1mer_frequencies = {k: float(v / n_kmers) for k, v in kmers_count.items()}
        return self._nt_1mer_frequencies

    def nt_2mer_frequencies(self) -> dict:
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

    def nucleotide_metrics(self) -> dict:
        """Computes a dictionary with all metrics for a DNA sequence"""

        sequence_metrics = {
            "nt_length": self.length,
            "pur_pyr_transition_freq": self.purine_pyrimidine_transition_freq(),
        }

        # Must prepend with "nt_" because code overlaps with aas
        for nt, count in self.nt_1mer_frequencies().items():  # 2 variables
            sequence_metrics["nt_{}".format(nt)] = float(count)

        return sequence_metrics
