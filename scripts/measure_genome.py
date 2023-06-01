#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict
from glob import glob
from itertools import chain, product
import json
import logging
import multiprocessing
from pathlib import Path
import subprocess

from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

from utils import fasta_iter

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

class Genome():

    def __init__(self, protein_faa_filepath : Path, fna_filepath : Path):
        self.faa_filepath = str(protein_faa_filepath)
        self.fna_filepath = str(fna_filepath)
        self.prefix = '.'.join(self.faa_filepath.split('/')[-1].split('.')[:-1])
        self.protein_statistics = None

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
                    # to-do: add other statistics below
                    self.protein_statistics[protein_id] = {
                        'length' : protein_calc.length,
                        'pi' : protein_calc.isoelectric_point(),
                        'gravy' : protein_calc.gravy(),
                        'zc' : protein_calc.zc(),
                        'nh2o' : protein_calc.nh2o(),
                        'aa_frequencies' : protein_calc.aa_frequencies(),
                        'f_ivywrel' : protein_calc.fraction_thermostable_ivywrel(),
                        'extra_soluble' : protein_calc.is_soluble_extracellular_heuristic(),
                    }
                    
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
        pis = self._nonnull(statistics_lists['pi'])
        proteome_statistics['histogram_pi'] = self._bin_midpoints(pis, bins=np.linspace(0, 14, 141))
        proteome_statistics['ratio_acidic_pis'] = len(pis[pis < 7]) / len(pis[pis >= 7])
        for aa, count in Counter(chain(*statistics_lists['aa_frequencies'])).items():
            proteome_statistics[f'aa_{aa.lower()}'] = count

        # means
        proteome_statistics['mean_protein_length'] = np.mean(self._nonnull(statistics_lists['length']))
        proteome_statistics['mean_pi'] = np.mean(pis)
        proteome_statistics['mean_gravy'] = np.mean(self._nonnull(statistics_lists['gravy']))
        proteome_statistics['mean_zc'] = np.mean(self._nonnull(statistics_lists['zc']))
        proteome_statistics['mean_nh2o'] = np.mean(self._nonnull(statistics_lists['nh2o']))
        proteome_statistics['mean_f_ivywrel'] = np.mean(self._nonnull(statistics_lists['f_ivywrel']))

        # weighted means
        proteome_statistics['weighted_mean_f_ivywrel'] = self._length_weighted(statistics_lists, 'f_ivywrel')
        proteome_statistics['weighted_mean_zc'] = self._length_weighted(statistics_lists, 'zc')
        proteome_statistics['weighted_mean_nh2o'] = self._length_weighted(statistics_lists, 'nh2o')
        proteome_statistics['weighted_mean_gravy'] = self._length_weighted(statistics_lists, 'gravy')

        return proteome_statistics

    def compute_genome_statistics(self):
        """
        Returns a dictionary of genome-wide statistics on nucleotide
        content, currently only k-mer counts
        """
        genome_statistics = {}
        with open(self.fna_filepath, 'r') as fh:
            genome = ''
            for header, sequence in fasta_iter(fh):
                genome += 'NN' + sequence
            nucleotide_calc = Nucleotide(genome)
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

        logger.info("{}: Collecting protein statistics".format(self.prefix))
        self.genomic_statistics['all'] = self.compute_proteome_statistics()
        self.genomic_statistics['all'].update(self.compute_genome_statistics())

        # Extracellular and soluble proteins
        extra_sol_proteins = set()
        for protein, stats in self.measure_protein_statistics().items():
            if stats['extra_soluble'] == True:
                extra_sol_proteins.add(protein)
        self.genomic_statistics['extracellular_soluble'] = self.compute_proteome_statistics(subset_proteins=extra_sol_proteins)

        return self.genomic_statistics

    def _nonnull(self, list_ : list):
        """Returns array without NaN values"""
        X = np.array(list_)
        return X[~np.isnan(X)]

    def _length_weighted(self, statistics_lists : dict, property : str) -> float:
        sum_ = np.sum(self._nonnull([f * l for f, l in zip(statistics_lists[property], statistics_lists['length'])]))
        sum_weights = np.sum(self._nonnull(statistics_lists['length']))
        if sum_weights > 0:
            return sum_ / sum_weights
        else:
            return np.nan

    def _bin_midpoints(self, data : np.array, bins: np.array):
        """Returns counts per bin keyed by bin midpoint"""
        bin_counts = []
        bin_midpoints = []
        counts = Counter(np.digitize(data, bins))
        for bin_idx in range(len(bins) - 1):
            bin_counts.append(counts.get(bin_idx, 0))
            bin_midpoints.append(round(np.mean([bins[bin_idx], bins[bin_idx + 1]]), 3))
        return dict(zip(bin_midpoints, bin_counts))

def _mapping_wrapper(paths):
    faa_path, fna_path = paths
    prefix = str(faa_path).split('/')[-1].replace('_protein.faa', '')
    return prefix, Genome(protein_faa_filepath=faa_path, fna_filepath=fna_path).collect_genomic_statistics()

def _format_pathlist_input(txt_file : str) -> list:
    pathlist = []
    with open(txt_file) as fh:
        for line in fh.readlines():
            path = line.strip()
            if Path(path).is_file():
                pathlist.append(path)
    return list(set(pathlist))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='GenomicProperties',
                    description='Computes statistics from genomes supplied in FASTA format'
                    )
    
    parser.add_argument('-faas', help='Path to subdirectory with proteins as amino acids in FASTA format, suffix .faa')
    parser.add_argument('-fnas', help='Path to subdirectory with proteins as nucleotides in FASTA format, suffix .fna')
    parser.add_argument('-p', default=4, help='Number of parallel processes', required=False)
    parser.add_argument('-o', '--output', default='genomic_properties.json', help='Output file name, default fasta_prefix.json')

    args = parser.parse_args()
    
    faa_pathlists = {str(path).split('/')[-1].replace('.faa', '') : path for path in Path(args.faas).rglob('*.faa')}
    fna_pathlists = {str(path).split('/')[-1].replace('.fna', '') : path for path in Path(args.fnas).rglob('*.fna')}

    pathlist = []
    for genome, faa_path in faa_pathlists.items():
        fna_path = fna_pathlists.get(genome, None)
        if fna_path:
            pathlist.append((faa_path, fna_path))
    
    output_json = str(args.output)
    workers = int(args.p)

    if workers is None:
        workers = multiprocessing.cpu_count() - 1

    logger.info("Measuring {} genomes with {} CPUs".format(len(pathlist), workers))
    filepath_gen = ((Path(faa_path), Path(fna_path)) for faa_path, fna_path in pathlist)
    with multiprocessing.Pool(workers) as p:
        pipeline_gen = p.map(_mapping_wrapper, filepath_gen)
        genomic_properties = dict(pipeline_gen)
    logger.info("Measured {} genomes".format(len(genomic_properties.keys())))
    
    json.dump(genomic_properties, open(output_json, 'w'))