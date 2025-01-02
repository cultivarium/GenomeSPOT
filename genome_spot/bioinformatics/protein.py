"""Class to compute info about primary sequences of proteins.

Information about primary sequences refers to what can be learned from 
the sequence alone independent of the gene's function or secondary or 
tertiary structure, for example the proportion of residues that are 
acidic.
"""

from typing import Dict

import numpy as np
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint

from ..helpers import count_kmers
from .signal_peptide import SignalPeptideHMM


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

# Dick et al. (2020)
# nH2O of the amino acid monomer minus one to get the amino acid residue
NH2O_QEC = {
    "A": 0.6 - 1,
    "C": 0.0 - 1,
    "D": -0.2 - 1,
    "E": 0.0 - 1,
    "F": -2.2 - 1,
    "G": 0.4 - 1,
    "H": -1.8 - 1,
    "I": 1.2 - 1,
    "K": 1.2 - 1,
    "L": 1.2 - 1,
    "M": 0.4 - 1,
    "N": -0.2 - 1,
    "P": 0.0 - 1,
    "Q": 0.0 - 1,
    "R": 0.2 - 1,
    "S": 0.6 - 1,
    "T": 0.8 - 1,
    "V": 1.0 - 1,
    "W": -3.8 - 1,
    "Y": -2.2 - 1,
}

# Dick et al. 2020
# Zc multiplied by the number of carbons of each amino acid
WEIGHTED_ZC = {
    "A": 0,
    "C": 2,
    "D": 4,
    "E": 2,
    "F": -4,
    "G": 2,
    "H": 4,
    "I": -6,
    "K": -4,
    "L": -6,
    "M": -2,
    "N": 4,
    "P": -2,
    "Q": 2,
    "R": 2,
    "S": 2,
    "T": 0,
    "V": -4,
    "W": -2,
    "Y": -2,
}

# Dick et al. 2020
# Number of carbons in each amino acid
CARBON_NUMBER = {
    "A": 3,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 9,
    "G": 2,
    "H": 6,
    "I": 6,
    "K": 6,
    "L": 6,
    "M": 5,
    "N": 4,
    "P": 5,
    "Q": 5,
    "R": 6,
    "S": 3,
    "T": 4,
    "V": 5,
    "W": 11,
    "Y": 9,
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


class Protein:
    """Calculations on a protein sequence.

    When counting amino acid frequencies, the initial
    Met amino acid (assumed to be present) is removed
    so that changes in protein length do not affect amino
    acid frequencies - e.g. shorter proteins incease Met
    frequency.

    Typical usage:
    ```
    protein_calc = Protein(protein_sequence)
    protein_metrics = protein_calc.protein_metrics() # all metrics
    pi = protein_calc.isoelectric_point() # individual metric
    ```
    """

    def __init__(
        self,
        protein_sequence: str,
        remove_signal_peptide: bool = True,
    ):
        """
        Args:
            protein_sequence: Amino acid sequence of one protein
        """
        self.sequence = self._format_protein_sequence(protein_sequence)
        self.length = len(self.sequence)
        self.start_pos = 1  # remove n-terminal Met
        self._aa_1mer_frequencies = None
        self._aa_2mer_frequencies = None
        self.signal_peptide_model = SignalPeptideHMM()
        self.remove_signal_peptide = remove_signal_peptide

    def _format_protein_sequence(self, protein_sequence: str) -> str:
        """Returns a formatted amino acid sequence"""
        return "".join([aa for aa in protein_sequence.strip().upper() if aa in STANDARD_AMINO_ACIDS])

    def aa_1mer_frequencies(self) -> Dict[str, float]:
        """Returns count of every amino acid ignoring start methionine"""
        if self._aa_1mer_frequencies is None:
            trimmed_sequence = self.sequence[self.start_pos :]
            if len(trimmed_sequence) >= 1:
                self._aa_1mer_frequencies = {
                    k: float(v / len(trimmed_sequence)) for k, v in count_kmers(trimmed_sequence, k=1).items()
                }
            else:
                self._aa_1mer_frequencies = {}
        return self._aa_1mer_frequencies

    def aa_2mer_frequencies(self) -> Dict[str, float]:
        """Returns count of every amino acid ignoring start methionine"""
        if self._aa_2mer_frequencies is None:
            if len(self.sequence[self.start_pos :]) >= 2:
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

    def gravy(self) -> float:
        """Compute the hydrophobicity as the
        Grand Average of Hydropathy (GRAVY)
        """
        if self.length > 0:
            return np.mean([HYDROPHOBICITY[aa] for aa in self.sequence[self.start_pos :]])
        else:
            return np.nan

    def zc(self) -> float:
        """Computes average carbon oxidation state (Zc) of a
        protein based on a dictionary of amino acids.
        """
        if self.length > 0:
            seq = self.sequence[self.start_pos :]
            return sum([WEIGHTED_ZC[s] for s in seq]) / sum([CARBON_NUMBER[s] for s in seq])
        else:
            return np.nan

    def nh2o(self) -> float:
        """Computes stoichiometric hydration state (nH2O) of a
        protein based on a dictionary of amino acids.
        """
        if self.length > 0:
            return sum([NH2O_QEC[s] for s in self.sequence[self.start_pos :]]) / self.length
        else:
            return np.nan

    def thermostable_freq(self) -> float:
        """Thermostable residues reported by:
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030005
        """
        if self.length > 0:
            return sum([v for k, v in self.aa_1mer_frequencies().items() if k in THERMOSTABLE_RESIDUES])
        else:
            return np.nan

    def protein_metrics(self) -> dict:
        """Computes a dictionary with all metrics for a protein"""

        (
            is_exported,
            signal_end_index,
        ) = self.signal_peptide_model.predict_signal_peptide(self.sequence)
        if self.remove_signal_peptide is True:
            self.start_pos = signal_end_index + 1
            # signal peptide should not be entire length
            if self.start_pos > self.length:
                self.start_pos = 1
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
        for aa in STANDARD_AMINO_ACIDS:
            freq = self.aa_1mer_frequencies().get(aa, 0)
            sequence_metrics["aa_{}".format(aa)] = freq

        return sequence_metrics
