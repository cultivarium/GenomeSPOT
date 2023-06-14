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
import json
import logging
from pathlib import Path
import subprocess as sp
from tempfile import NamedTemporaryFile

import docker
import numpy as np

from helpers import fasta_iter, nonnull, bin_midpoints
from primary_sequences import Protein, DNA

def _get_exported_proteins_with_deepsig(protein_fasta, organism_type) -> set:
    """Uses a preexisting Docker container to run DeepSig.
    The protein FASTA must be located under the current working
    directory.

    Args:
        protein_fasta: path to protein FASTA
        organism_type: either euk/gramp/gramn
    """

    # Run DeepSig
    if protein_fasta.endswith('.gz'):
        input_fasta = protein_fasta.replace('.gz', '')
        with open(input_fasta, 'w') as fhout:
            with io.TextIOWrapper(io.BufferedReader(gzip.open(protein_fasta, 'r'))) as fh:
                for line in fh.readlines():
                    fhout.write(line)
    else:
        input_fasta = protein_fasta
    temp_output = input_fasta + '.deepsig.gff3'
    client = docker.from_env()
    local_bind = docker.types.Mount(
        source=str(Path.cwd()), 
        target='/home/cultivarium/files/deepsig/data', 
        type='bind')
    client.containers.run(
        "deepsig:latest", 
        f"-f data/{input_fasta} -o data/{temp_output} -k {organism_type}", 
        auto_remove=True, 
        mounts=[local_bind])
    
    # Parse output
    exported_proteins = set()
    with open(temp_output, 'r') as fh:
        for line in fh.readlines():
            entry = line.strip().split('\t')
            protein_id = entry[0]
            signal_prediction = entry[2]
            if signal_prediction == 'Signal peptide':
                exported_proteins.add(protein_id)
    
    # Remove intermediates
    if protein_fasta.endswith('.gz'):
        Path(input_fasta).unlink()
    Path(temp_output).unlink()

    return exported_proteins

class Genome():

    def __init__(self, protein_filepath : Path, contig_filepath : Path, organism_type : str):
        self.faa_filepath = str(protein_filepath)
        self.fna_filepath = str(contig_filepath)
        self.organism_type = str(organism_type).replace('arch', 'gramn')
        self.prefix = '.'.join(self.faa_filepath.split('/')[-1].split('.')[:-1])
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
            if self.faa_filepath.endswith('.gz'):
                fh = io.TextIOWrapper(io.BufferedReader(gzip.open(self.faa_filepath, 'r')))
            else:
                fh = open(self.faa_filepath, 'r')
            fh.seek(0)
            for header, sequence in fasta_iter(fh):
                protein_id = header.split(' ')[0]
                self._protein_data[protein_id] = Protein(sequence).protein_metrics()
            fh.close()
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

        protein_statistics['total_proteins'] = len(values_dict['length'])
        protein_statistics['total_protein_length'] = int(np.sum(values_dict['length']))

        # distributions
        pis = np.array(values_dict['pi'])
        protein_statistics['pis_acidic'] = float(np.sum((pis < 5.5)) / len(pis))
        protein_statistics['pis_neutral'] = float(np.sum(((pis >= 5.5) & (pis < 8.5))) / len(pis))
        protein_statistics['pis_basic'] = float( np.sum((pis >= 8.5)) / len(pis))

        # means
        protein_statistics['mean_pi'] = float(np.mean(pis))
        protein_statistics['mean_gravy'] = float(np.mean(values_dict['gravy']))
        protein_statistics['mean_zc'] = float(np.mean(values_dict['zc']))
        protein_statistics['mean_nh2o'] = float(np.mean(values_dict['nh2o']))
        protein_statistics['mean_thermostable_freq'] = float(np.mean(values_dict['thermostable_freq']))
        protein_statistics['mean_protein_length'] = float(np.mean(values_dict['length']))

        # ratios and proportion
        arg = np.mean(values_dict.get('aa_R', 0.))
        lys = np.mean(values_dict.get('aa_K', 0.))
        if arg + lys > 0:
            protein_statistics['proportion_R_RK'] = float(arg / (arg + lys))

        # amino acid k-mer frequencies
        for variable, values in values_dict.items():
            if variable.startswith('aa_'):
                protein_statistics[variable] = float(np.mean(values))

        return protein_statistics

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
            exported_proteins = _get_exported_proteins_with_deepsig(self.faa_filepath, self.organism_type)
            self.exported_proteins = exported_proteins
            mean_hydrophobicity = np.mean([v['gravy'] for k, v in self.protein_data().items() if v['gravy'] is not None])

            self._protein_localization = {}
            for protein in self.protein_data().keys():
                hydrophobicity = self.protein_data()[protein]['gravy']
                if (hydrophobicity - mean_hydrophobicity) >= Protein.DIFF_HYDROPHOBICITY_MEMBRANE:
                    self._protein_localization[protein] = 'membrane'
                else:                        
                    if protein in exported_proteins:
                        self._protein_localization[protein] = 'extra_soluble'
                    else:
                        self._protein_localization[protein] = 'intra_soluble'

        return self._protein_localization

    def compute_dna_statistics(self):
        """Returns a dictionary of genome-wide statistics on nucleotide
        content, currently only k-mer counts
        """
        genome_statistics = {}
        if self.fna_filepath.endswith('.gz'):
            fh = io.TextIOWrapper(io.BufferedReader(gzip.open(self.fna_filepath, 'r')))
        else:
            fh = open(self.fna_filepath, 'r')
        fh.seek(0)
        genome = ''
        for header, sequence in fasta_iter(fh):
            genome += 'NN' + sequence
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
        extracellular_soluble = {protein for protein, locale in localization.items() if locale == 'extra_soluble'}
        intracellular_soluble = {protein for protein, locale in localization.items() if locale == 'intra_soluble'}
        membrane = {protein for protein, locale in localization.items() if locale == 'membrane'}

        logging.info("{}: Collecting genome statistics".format(self.prefix))
        self.genomic_statistics['all'] = self.compute_dna_statistics()

        logging.info("{}: Collecting protein statistics".format(self.prefix))
        self.genomic_statistics['all'].update(self.compute_protein_statistics())
        self.genomic_statistics['all']['protein_coding_density']  = 3 * self.genomic_statistics['all']['total_protein_length'] / self.genomic_statistics['all']['nt_length']
        if extracellular_soluble:
            self.genomic_statistics['extracellular_soluble'] = self.compute_protein_statistics(subset_proteins=extracellular_soluble)
        if intracellular_soluble:
            self.genomic_statistics['intracellular_soluble'] = self.compute_protein_statistics(subset_proteins=intracellular_soluble)
        if membrane:
            self.genomic_statistics['membrane'] = self.compute_protein_statistics(subset_proteins=membrane)

        return self.genomic_statistics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='MeasureGenome',
                    description='Computes statistics from a genome'
                    )
    
    parser.add_argument('-c', '--contigs', help="Path to a gzipped FASTA of contigs")
    parser.add_argument('-p', '--proteins', help="Path toa gzipped FASTA of proteins")
    parser.add_argument('-t', '--organism-type', help="Type of organism {gramn/gramp/arch}; if arch defaults to gramp")
    parser.add_argument('-o', '--output', help='Output file name, default <genome_prefix>.json')

    args = parser.parse_args()
    
    if args.output:
        output = str(args.output)
    else:
        output = '.'.join(str(args.contigs).split('.')[:-1]) + '.json'

    genome_features = Genome(
        protein_filepath=args.proteins, 
        contig_filepath=args.contigs, 
        organism_type=args.organism_type
        ).genome_metrics()
    
    json.dump(genome_features, open(output, 'w'))