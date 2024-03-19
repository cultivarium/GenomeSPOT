"""TaxonomyGTDB is a helper class to use the Genome Taxonomy Database (GTDB) taxonomy."""

import gzip
import io
import subprocess as sp
from collections import Counter
from pathlib import Path
from typing import (
    Optional,
    Tuple,
    Union,
)

import numpy as np


class TaxonomyGTDB:
    """
    Class for taxonomy operations using the taxonomy
    from the Genome Taxonomy Database (gtdb.ecogenomic.org/)

    Typical usage:
        ```
        from taxonomy import TaxonomyGTDB
        taxonomy = TaxonomyGTDB()
        genome_taxonomy = taxonomy.taxonomy_dict['GCA_000979555']
        taxon = taxonomy.taxa_of_genomes(['GCA_000979555', 'GCA_017565965'], 'family')
        ```

    Args:
        taxonomy_filenames: list of taxonomy files. If not provided,
            will download files from GTDB if files are not in path.

    """

    def __init__(self, taxonomy_filenames: Optional[list] = None):
        self.indices = {
            "domain": 0,
            "phylum": 1,
            "class": 2,
            "order": 3,
            "family": 4,
            "genus": 5,
            "species": 6,
        }
        self.taxonomy_filenames = self.download_taxonomy_files(taxonomy_filenames)
        self.taxonomy_dict = self.make_taxonomy_dict()

    def download_taxonomy_files(self, taxonomy_filenames: Optional[list] = None):
        """Downloads taxonomy files if desired files not already in path"""
        if taxonomy_filenames is None:
            desired_files = ["ar53_taxonomy.tsv.gz", "bac120_taxonomy.tsv.gz"]
            if any([Path(file).exists() is False for file in desired_files]):
                for file in desired_files:
                    cmd = f"wget https://data.gtdb.ecogenomic.org/releases/latest/{file}"
                    print(cmd)
                    sp.run(cmd, shell=True, check=True)
            return desired_files
        else:
            return taxonomy_filenames

    def make_taxonomy_dict(self) -> dict:
        """
        Load a taxonomic file into a dict keyed by genome accession.

        For example:
            {'GCA_016456235': ('Bacteria',
                            'Pseudomonadota',
                            'Gammaproteobacteria',
                            'Enterobacterales',
                            'Enterobacteriaceae',
                            'Escherichia',
                            'Escherichia coli'),...}

        """

        taxonomy_dict = {}

        for filename in self.taxonomy_filenames:
            if filename.endswith(".gz"):
                fh = gzip.open(filename, "rt")
            else:
                fh = open(filename, "r")
            fh.seek(0)
            for line in fh.readlines():
                gtdb_accession, taxstring = line.strip().split("\t")
                ncbi_accession = self.convert_gtdb_to_ncbi(gtdb_accession, make_genbank=True, remove_version=True)
                _, taxonomy = self.format_taxonomy_as_tuple(taxstring)
                taxonomy_dict[ncbi_accession] = taxonomy
            fh.close()
        return taxonomy_dict

    def convert_gtdb_to_ncbi(self, accession: str, make_genbank: bool = True, remove_version: bool = True) -> str:
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
            ncbi_accession = ncbi_accession.replace("GCF_", "GCA_")
        if remove_version:
            ncbi_accession = ncbi_accession[:-2]
        return ncbi_accession

    def format_taxonomy_as_tuple(self, taxstring: str) -> Tuple:
        """Convert GTDB taxstring to a dictionary.

        Args:
            taxstring: A GTDB taxstring in the following format:
                d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;...
        Returns:
            taxonomy_dict: A dictionary is keyed by the following ranks: domain, phylum, class,
                order, family, genus, and species.
        """
        ABBREV = {
            "d": "domain",
            "p": "phylum",
            "c": "class",
            "o": "order",
            "f": "family",
            "g": "genus",
            "s": "species",
        }

        levels = []
        names = []
        for level in taxstring.strip().split(";"):
            abbrev, taxon = level.split("__")
            levels.append(ABBREV[abbrev])
            names.append(taxon)
        return tuple(levels), tuple(names)

    def taxonomy_dict_at_taxlevel(self, taxlevel: str) -> dict:
        """Returns taxonomy at the specified level"""
        index = self.indices[taxlevel]
        return {k: v[index] for k, v in self.taxonomy_dict.items()}

    def measure_diversity(self, query_rank: str, diversity_rank: str, subset_genomes: Optional[list] = None) -> dict:
        """Counts the number of taxa at rank `diversity_rank`
        under each taxon of the `query_rank` for a set of genomes.
        """
        query_index = self.indices[query_rank]
        diversity_index = self.indices[diversity_rank]
        ancestor_dict = {}
        if subset_genomes:
            for genome in subset_genomes:
                taxonomy = self.taxonomy_dict.get(genome, None)
                if taxonomy:
                    ancestor_dict[taxonomy[diversity_index]] = taxonomy[query_index]
        else:
            for genome, taxonomy in self.taxonomy_dict.items():
                ancestor_dict[taxonomy[diversity_index]] = taxonomy[query_index]

        return dict(Counter(ancestor_dict.values()))

    def taxa_of_genomes(self, genomes: Union[list, set], taxonomic_level: str):
        """Get taxa for a set of genomes at the specified level"""
        taxa = set()
        for genome in genomes:
            taxonomy = self.taxonomy_dict.get(genome, None)
            if taxonomy:
                taxa.add(taxonomy[self.indices[taxonomic_level]])
        return sorted(taxa)

    def genomes_in_taxa(self, taxa: list, taxonomic_level: str):
        """Get all genomes for a set of taxa, must specify level"""
        genomes = set()
        for genome, taxonomy in self.taxonomy_dict.items():
            taxon = taxonomy[self.indices[taxonomic_level]]
            if taxon in taxa:
                genomes.add(genome)
        return sorted(genomes)
