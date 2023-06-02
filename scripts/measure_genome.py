#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict
from itertools import chain
import json
import logging
from pathlib import Path
import subprocess

import numpy as np

from helpers import fasta_iter, nonnull, bin_midpoints
from primary_sequences import Protein, DNA

class Genome():

    def __init__(self, protein_filepath : Path, contig_filepath : Path):
        self.faa_filepath = str(protein_filepath)
        self.fna_filepath = str(contig_filepath)
        self.prefix = '.'.join(self.faa_filepath.split('/')[-1].split('.')[:-1])
        self._protein_data = None
    
    def protein_data(self):
        """
        Returns a dictionary of properties for each protein.

        Association of properties to every protein allows statistics 
        to be computed jointly with multiple values, such as weighting
        a statistic by protein length.
        """
        if self._protein_data is None:
            self._protein_data = {}
            with open(self.faa_filepath, 'r') as fh:
                for header, sequence in fasta_iter(fh):
                    protein_id = header.split(' ')[0]
                    self._protein_data[protein_id] = Protein(sequence).protein_metrics()
                    
        return self._protein_data

    def compute_protein_statistics(self, subset_proteins : set=None) -> dict:
        """
        Returns a dictionary of genome-wide statistics, based on 
        measurements, to be used for downstream analyses
        """
        protein_statistics = {}

        if subset_proteins:
            values_by_protein = {k : self.protein_data()[k] for k in subset_proteins}
        else:
            values_by_protein = self.protein_data()
        
        values_dict = defaultdict(list)
        for protein, stats in values_by_protein.items():
            for key, value in stats.items():
                if value:
                    values_dict[key].append(value)

        protein_statistics['sum_protein_length'] = np.sum(values_dict['length'])

        # amino acid k-mer frequencies
        for variable, values in values_dict.items():
            if variable.startswith('aa_'):
                protein_statistics['mean_{}'.format(variable)] = np.mean(values)

        # distributions
        pis = np.array(values_dict['pi'])
        protein_statistics['pis_acidic'] = np.sum((pis < 5.5)) / len(pis)
        protein_statistics['pis_neutral'] = np.sum(((pis >= 5.5) & (pis < 8.5))) / len(pis)
        protein_statistics['pis_basic'] =  np.sum((pis >= 8.5)) / len(pis)

        # means
        protein_statistics['mean_pi'] = np.mean(pis)
        protein_statistics['mean_gravy'] = np.mean(values_dict['gravy'])
        protein_statistics['mean_zc'] = np.mean(values_dict['zc'])
        protein_statistics['mean_nh2o'] = np.mean(values_dict['nh2o'])
        protein_statistics['mean_thermostable_freq'] = np.mean(values_dict['thermostable_freq'])
        protein_statistics['mean_protein_length'] = np.mean(values_dict['length'])

        # ratios and proportion
        arg = protein_statistics.get('mean_aa_R', 0.) 
        lys = protein_statistics.get('mean_aa_K', 0.)
        if arg + lys > 0:
            protein_statistics['proportion_R_RK'] = arg / (arg + lys)

        return protein_statistics


    def _call_signal_pred(self):
        #
        return {}

    def identify_protein_localization(self):

        # run signal pred
        self._call_signal_pred()

        # get hydrophobicity

        # assign localization: intra soluble, membrane, extra soluble

        return {}

    def compute_dna_statistics(self):
        """Returns a dictionary of genome-wide statistics on nucleotide
        content, currently only k-mer counts
        """
        genome_statistics = {}
        with open(self.fna_filepath, 'r') as fh:
            genome = ''
            for header, sequence in fasta_iter(fh):
                genome += 'NN' + sequence

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
        localization = self.identify_protein_localization()
        extracellular_soluble = {protein for protein, locale in localization.items() if locale == 'extra_soluble'}
        intracellular_soluble = {protein for protein, locale in localization.items() if locale == 'intra_soluble'}

        logging.info("{}: Collecting genome statistics".format(self.prefix))
        self.genomic_statistics['all'] = self.compute_dna_statistics()

        logging.info("{}: Collecting protein statistics".format(self.prefix))
        self.genomic_statistics['all'].update(self.compute_protein_statistics())
        self.genomic_statistics['extracellular_soluble'] = self.compute_protein_statistics(subset_proteins=extracellular_soluble)
        self.genomic_statistics['intracellular_soluble'] = self.compute_protein_statistics(subset_proteins=intracellular_soluble)
        self.genomic_statistics['all']['protein_coding_density']  = 3 * self.genomic_statistics['all']['sum_protein_length'] / self.genomic_statistics['all']['nt_length']

        return self.genomic_statistics





if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='MeasureGenome',
                    description='Computes statistics from a genome'
                    )
    
    parser.add_argument('-c', '--contigs', help="Path to a genome's contigs in FASTA format")
    parser.add_argument('-p', '--proteins', help="Path to a genome's proteins in FASTA format")
    parser.add_argument('-o', '--output', help='Output file name, default <genome_prefix>.json')

    args = parser.parse_args()
    
    genome_calc = Genome(protein_filepath=args.proteins, contig_filepath=args.contigs,)
    genome_features = genome_calc.features()
    json.dump(genome_features, open(str(args(output)), 'w'))