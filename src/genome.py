#!/usr/bin/env python3

"""
Measures properties of a genome (fasta + protein-fasta).

Example usage:

```shell
python3 measure_genome.py -t arch -c data/genomes/genbank/archaea/GCA_000007185.1/GCA_000007185.1_ASM718v1_genomic.fna.gz -p data/genomes/genbank/archaea/GCA_000007185.1/GCA_000007185.1_ASM718v1_protein.faa.gz 
```
"""

import argparse
from collections import defaultdict
import gzip
import io
import joblib
import json
import logging
from pathlib import Path
import subprocess as sp

import numpy as np

from helpers import fasta_iter
from protein import Protein
from dna import DNA
from signal_peptide import SignalPeptideHMM


class Genome:
    def __init__(
        self,
        contig_filepath: str,
        protein_filepath: str,
    ):
        self.fna_filepath = str(contig_filepath)
        self.faa_filepath = str(protein_filepath)
        if Path(self.fna_filepath).exists() is False:
            raise ValueError(f"Input file {self.fna_filepath} does not exist")
        if Path(self.faa_filepath).exists() is False:
            raise ValueError(f"Input file {self.faa_filepath} does not exist")
        self.prefix = self.fna_filepath.split("/")[-1]
        self._protein_data = None
        self._protein_localization = None

    def protein_data(self):
        """
        Returns a dictionary of properties for each protein.

        Association of properties to every protein allows statistics
        to be computed jointly with multiple values, such as weighting
        a statistic by protein length.
        """
        if self._protein_data is None:
            self._protein_data = {}
            if self.faa_filepath.endswith(".gz"):
                fh = io.TextIOWrapper(
                    io.BufferedReader(gzip.open(self.faa_filepath, "r"))
                )
            else:
                fh = open(self.faa_filepath, "r")
            fh.seek(0)
            for header, sequence in fasta_iter(fh):
                protein_id = header.split(" ")[0]
                self._protein_data[protein_id] = Protein(
                    protein_sequence=sequence,
                    remove_signal_peptide=True,
                ).protein_metrics()
            fh.close()
        return self._protein_data

    def compute_protein_statistics(self, subset_proteins: set = None) -> dict:
        """
        Returns a dictionary of genome-wide statistics, based on
        measurements, to be used for downstream analyses
        """
        protein_statistics = {}

        if subset_proteins:
            values_by_protein = {k: self.protein_data()[k] for k in subset_proteins}
        else:
            values_by_protein = self.protein_data()

        values_dict = defaultdict(list)
        for protein, stats in values_by_protein.items():
            for key, value in stats.items():
                if value:
                    values_dict[key].append(value)

        protein_statistics["total_proteins"] = len(values_dict["length"])
        protein_statistics["total_protein_length"] = int(np.sum(values_dict["length"]))

        # distributions
        pis = np.array(values_dict["pi"])
        protein_statistics["pis_acidic"] = float(np.sum((pis < 5.5)) / len(pis))
        protein_statistics["pis_neutral"] = float(
            np.sum(((pis >= 5.5) & (pis < 8.5))) / len(pis)
        )
        protein_statistics["pis_basic"] = float(np.sum((pis >= 8.5)) / len(pis))
        step = 1
        for i in range(3, 12, step):
            protein_statistics["pis_{}_{}".format(i, i + step)] = len(
                [pi for pi in pis if pi >= i and pi < (i + 1)]
            ) / len(pis)

        # means

        protein_statistics["mean_pi"] = float(np.mean(pis))
        protein_statistics["mean_gravy"] = float(np.mean(values_dict["gravy"]))
        protein_statistics["mean_zc"] = float(np.mean(values_dict["zc"]))
        protein_statistics["mean_nh2o"] = float(np.mean(values_dict["nh2o"]))
        protein_statistics["mean_protein_length"] = float(
            np.mean(values_dict["length"])
        )
        protein_statistics["mean_thermostable_freq"] = self._length_weighted_average(
            values_dict["thermostable_freq"], values_dict["length"]
        )

        # ratios and proportion
        zeros_list = [0] * len(values_dict["length"])
        arg = self._length_weighted_average(
            values_dict.get("aa_R", zeros_list), values_dict["length"]
        )
        lys = self._length_weighted_average(
            values_dict.get("aa_K", zeros_list), values_dict["length"]
        )
        if arg + lys > 0:
            protein_statistics["proportion_R_RK"] = float(arg / (arg + lys))

        # amino acid k-mer frequencies
        for variable, values in values_dict.items():
            if variable.startswith("aa_"):
                protein_statistics[variable] = self._length_weighted_average(
                    values, values_dict["length"]
                )

        return protein_statistics

    def _length_weighted_average(self, values, lengths):
        return float(
            np.sum([length * val for length, val in zip(lengths, values)])
            / np.sum(lengths)
        )

    def protein_localization(self):
        """Localizes proteins to inside/outside/within the cell membrane.

        Extracellular proteins are identified with a signal peptide prediction
        software. Membrane proteins are identified with a heuristic of having a
        hydrophobicity of GRAVY > 0 and must have a signal peptide. Other proteins
        are intracellular soluble proteins.

        Returns:
            localization: dictionary of protein key with values either
                'membrane', 'extra_soluble', or 'intra_soluble'
        """

        if self._protein_localization is None:
            # exported_proteins = _get_exported_proteins_with_deepsig(self.faa_filepath, self.organism_type)
            # self.exported_proteins = exported_proteins
            mean_hydrophobicity = np.mean(
                [
                    v["gravy"]
                    for k, v in self.protein_data().items()
                    if v["gravy"] is not None
                ]
            )

            self._protein_localization = {}
            for protein in self.protein_data().keys():
                hydrophobicity = self.protein_data()[protein]["gravy"]
                if (
                    hydrophobicity - mean_hydrophobicity
                ) >= Protein.DIFF_HYDROPHOBICITY_MEMBRANE:
                    self._protein_localization[protein] = "membrane"
                else:
                    # Get localization
                    is_exported = self.protein_data()[protein]["is_exported"]
                    if is_exported is True:
                        self._protein_localization[protein] = "extra_soluble"
                    else:
                        self._protein_localization[protein] = "intra_soluble"

        return self._protein_localization

    def compute_dna_statistics(self):
        """Returns a dictionary of genome-wide statistics on nucleotide
        content, currently only k-mer counts
        """
        genome_statistics = {}
        if self.fna_filepath.endswith(".gz"):
            fh = io.TextIOWrapper(io.BufferedReader(gzip.open(self.fna_filepath, "r")))
        else:
            fh = open(self.fna_filepath, "r")
        fh.seek(0)
        genome = ""
        for header, sequence in fasta_iter(fh):
            genome += "NN" + sequence
        fh.close()

        nucleotide_calc = DNA(genome)
        genome_statistics.update(nucleotide_calc.nucleotide_metrics())
        return genome_statistics

    def genome_metrics(self) -> dict:
        """
        Computes statistics about the proteome for each genome on:
        1. All proteins, keyed by 'all'
        2. Extracellular soluble proteins, keyed by 'extracellular_soluble'
        """

        self.genomic_statistics = {}

        logging.info("{}: Identifying protein localization".format(self.prefix))
        localization = self.protein_localization()
        extracellular_soluble = {
            protein
            for protein, locale in localization.items()
            if locale == "extra_soluble"
        }
        intracellular_soluble = {
            protein
            for protein, locale in localization.items()
            if locale == "intra_soluble"
        }
        membrane = {
            protein for protein, locale in localization.items() if locale == "membrane"
        }

        logging.info("{}: Collecting genome statistics".format(self.prefix))
        self.genomic_statistics["all"] = self.compute_dna_statistics()

        logging.info("{}: Collecting protein statistics".format(self.prefix))
        self.genomic_statistics["all"].update(self.compute_protein_statistics())
        self.genomic_statistics["all"]["protein_coding_density"] = (
            3
            * self.genomic_statistics["all"]["total_protein_length"]
            / self.genomic_statistics["all"]["nt_length"]
        )
        if extracellular_soluble:
            self.genomic_statistics[
                "extracellular_soluble"
            ] = self.compute_protein_statistics(subset_proteins=extracellular_soluble)
        if intracellular_soluble:
            self.genomic_statistics[
                "intracellular_soluble"
            ] = self.compute_protein_statistics(subset_proteins=intracellular_soluble)
        if membrane:
            self.genomic_statistics["membrane"] = self.compute_protein_statistics(
                subset_proteins=membrane
            )

        self.genomic_statistics["diff_extra_intra"] = {}
        for key, val_extra in self.genomic_statistics["extracellular_soluble"].items():
            val_intra = self.genomic_statistics["intracellular_soluble"].get(
                key, np.nan
            )
            self.genomic_statistics["diff_extra_intra"][key] = val_extra - val_intra

        return self.genomic_statistics