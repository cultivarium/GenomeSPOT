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
            if self.length > 1:
                self._aa_1mer_frequencies = {
                    k: float(v / len(self.sequence[self.start_pos :]))
                    for k, v in count_kmers(self.sequence[self.start_pos :], k=1).items()
                }
            else:
                self._aa_1mer_frequencies = {}
        return self._aa_1mer_frequencies

    def aa_2mer_frequencies(self) -> Dict[str, float]:
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
        return sum([WEIGHTED_ZC[s] for s in self.sequence[self.start_pos :]]) / self.length

    def nh2o(self) -> float:
        """Computes stoichiometric hydration state (nH2O) of a
        protein based on a dictionary of amino acids.
        """
        return sum([NH2O_RQEC[s] for s in self.sequence[self.start_pos :]]) / self.length

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

        return sequence_metrics
