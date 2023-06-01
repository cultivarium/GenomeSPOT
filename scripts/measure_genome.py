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
        self.protein_statistics = None

    def _length_weighted(self, statistics_lists : dict, property : str) -> float:
        sum_ = np.sum(nonnull([f * l for f, l in zip(statistics_lists[property], statistics_lists['length'])]))
        sum_weights = np.sum(nonnull(statistics_lists['length']))
        if sum_weights > 0:
            return sum_ / sum_weights
        else:
            return np.nan
    
    def measure_protein_statistics(self):
        """
        Returns a dictionary of properties for each protein.

        Association of properties to every protein allows statistics 
        to be computed jointly with multiple values, such as weighting
        a statistic by protein length.
        """
        if self.protein_statistics is None:
            self.protein_statistics = {}
            with open(self.faa_filepath, 'r') as fh:
                for header, sequence in fasta_iter(fh):
                    protein_id = header.split(' ')[0]
                    protein_calc = Protein(sequence)
                    self.protein_statistics[protein_id] = protein_calc.protein_metrics()
                    
        return self.protein_statistics

    def compute_proteome_statistics(self, subset_proteins : set=None) -> dict:
        """
        Returns a dictionary of genome-wide statistics, based on 
        measurements, to be used for downstream analyses
        """
        proteome_statistics = {}

        if subset_proteins:
            protein_statistics_dict = {k : self.measure_protein_statistics()[k] for k in subset_proteins}
        else:
            protein_statistics_dict = self.measure_protein_statistics()
        
        statistics_lists = defaultdict(list)
        for protein, stats in protein_statistics_dict.items():
            for key, value in stats.items():
                statistics_lists[key].append(value)

        # distributions / frequencies
        pis = nonnull(statistics_lists['pi'])
        proteome_statistics['histogram_pi'] = bin_midpoints(pis, bins=np.linspace(0, 14, 141))
        proteome_statistics['ratio_acidic_pis'] = len(pis[pis < 7]) / len(pis[pis >= 7])
        for aa, count in Counter(chain(*statistics_lists['aa_frequencies'])).items():
            proteome_statistics[f'aa_{aa.lower()}'] = count

        # means
        proteome_statistics['mean_protein_length'] = np.mean(nonnull(statistics_lists['length']))
        proteome_statistics['mean_pi'] = np.mean(pis)
        proteome_statistics['mean_gravy'] = np.mean(nonnull(statistics_lists['gravy']))
        proteome_statistics['mean_zc'] = np.mean(nonnull(statistics_lists['zc']))
        proteome_statistics['mean_nh2o'] = np.mean(nonnull(statistics_lists['nh2o']))
        proteome_statistics['mean_f_ivywrel'] = np.mean(nonnull(statistics_lists['f_ivywrel']))

        # weighted means
        proteome_statistics['weighted_mean_f_ivywrel'] = self._length_weighted(statistics_lists, 'f_ivywrel')
        proteome_statistics['weighted_mean_zc'] = self._length_weighted(statistics_lists, 'zc')
        proteome_statistics['weighted_mean_nh2o'] = self._length_weighted(statistics_lists, 'nh2o')
        proteome_statistics['weighted_mean_gravy'] = self._length_weighted(statistics_lists, 'gravy')

        return proteome_statistics


    def _call_signal_pred(self):
        #
        return

    def identify_protein_localization(self):

        # run signal pred
        self._call_signal_pred()
        
        # get hydrophobicity

        # assign localization: intra soluble, membrane, extra soluble

        return 

    def compute_genome_statistics(self):
        """Returns a dictionary of genome-wide statistics on nucleotide
        content, currently only k-mer counts
        """
        genome_statistics = {}
        with open(self.fna_filepath, 'r') as fh:
            genome = ''
            for header, sequence in fasta_iter(fh):
                genome += 'NN' + sequence
                nucleotide_calc = DNA(genome)
                genome_statistics.update(nucleotide_calc.count_canonical_kmers(k=1))
                genome_statistics.update(nucleotide_calc.count_canonical_kmers(k=2))

        return genome_statistics

    def collect_genomic_statistics(self) -> dict:
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
        
        logging.info("{}: Collecting protein statistics".format(self.prefix))
        self.genomic_statistics['all'] = self.compute_proteome_statistics()
        self.genomic_statistics['intracellular_soluble'] = self.compute_proteome_statistics(subset_proteins=intracellular_soluble)
        self.genomic_statistics['extracellular_soluble'] = self.compute_proteome_statistics(subset_proteins=extracellular_soluble)

        logging.info("{}: Collecting genome statistics".format(self.prefix))
        self.genomic_statistics['all'].update(self.compute_genome_statistics())

        return self.genomic_statistics





if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='MeasureGenome',
                    description='Computes statistics from a genome'
                    )
    
    parser.add_argument('-c', '--contigs' help="Path to a genome's contigs in FASTA format")
    parser.add_argument('-p', '--proteins', help="Path to a genome's proteins in FASTA format")
    parser.add_argument('-o', '--output', help='Output file name, default <genome_prefix>.json')

    args = parser.parse_args()
    
    genome_calc = Genome(protein_filepath=args.proteins, contig_filepath=args.contigs,)
    genome_features = genome_calc.features()
    json.dump(genome_features, open(str(args(output)), 'w'))