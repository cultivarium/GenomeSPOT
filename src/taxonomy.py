"""
Classes to perform operations on a taxonomic basis:

    - TaxonomyGTDB: taxonomy helper
    - BalanceTaxa: balances a dataset using taxonomy
    - PartitionTaxa: partitions a dataset using taxonomy
"""

from collections import defaultdict, Counter
import subprocess as sp
from pathlib import Path
import gzip
import io

import numpy as np

RANDOM_SEED = 1111
rng = np.random.default_rng(RANDOM_SEED)


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

    def __init__(self, taxonomy_filenames: list = None):
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

    def download_taxonomy_files(self, taxonomy_filenames: list = None):
        """Downloads taxonomy files if desired files not already in path"""
        if taxonomy_filenames is None:
            desired_files = ["ar53_taxonomy.tsv.gz", "bac120_taxonomy.tsv.gz"]
            if any([Path(file).exists() is False for file in desired_files]):
                for file in desired_files:
                    cmd = (
                        f"wget https://data.gtdb.ecogenomic.org/releases/latest/{file}"
                    )
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
            # with open(filename, "r") as fh:
            fh = io.TextIOWrapper(io.BufferedReader(gzip.open(filename, "r")))
            fh.seek(0)
            for line in fh.readlines():
                gtdb_accession, taxstring = line.strip().split("\t")
                ncbi_accession = self.gtdb_accession_to_ncbi(
                    gtdb_accession, make_genbank=True, remove_version=True
                )
                _, taxonomy = self.gtdb_taxonomy_to_tuple(taxstring)
                taxonomy_dict[ncbi_accession] = taxonomy
            fh.close()
        return taxonomy_dict

    def gtdb_accession_to_ncbi(
        self, accession: str, make_genbank: bool = True, remove_version: bool = True
    ) -> str:
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

    def gtdb_taxonomy_to_tuple(self, taxstring: str) -> dict:
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

    def measure_diversity(
        self, query_rank: str, diversity_rank: str, subset_genomes: list = None
    ) -> dict:
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

    def taxa_of_genomes(self, genomes: list, taxonomic_level: str):
        """Get taxa for a set of genomes at the specified level"""
        taxa = set()
        for genome in genomes:
            taxonomy = self.taxonomy_dict.get(genome, None)
            if taxonomy:
                taxa.add(taxonomy[self.indices[taxonomic_level]])
        return taxa

    def genomes_in_taxa(self, taxa: list, taxonomic_level: str):
        """Get all genomes for a set of taxa, must specify level"""
        genomes = set()
        for genome, taxonomy in self.taxonomy_dict.items():
            taxon = taxonomy[self.indices[taxonomic_level]]
            if taxon in taxa:
                genomes.add(genome)
        return genomes


class BalanceTaxa:
    """
    Tool to balance a dataset - i.e. be less biased, compared to a reference.

    Balancing: The user specifies a number of taxa to keep, and taxa are
    randomly chosen with a higher likelihood of being chosen inversely
    proportional to the bias for that taxon in the dataset compared to the
    supplied taxonomic reference.

    Typical usage:
        ```
        from taxonomy import BalanceTaxa, TaxonomyGTDB
        taxonomy = TaxonomyGTDB()
        balancer = BalanceTaxa(taxonomy=taxonomy)
        balanced_genomes = balancer.balance_dataset(
            genomes=genomes,
            proportion_to_keep=0.5,
            diversity_rank="species"
        )
        ```
    """

    def __init__(
        self,
        taxonomy: TaxonomyGTDB,
    ):
        self.taxonomy = taxonomy

    def balance_dataset(
        self,
        genomes: list,
        proportion_to_keep: float = None,
        diversity_rank: str = "species",
    ):
        """
        Balance taxonomic groups by removing overrepresented taxa:
        observe the distribution of available
        genomes by taxon relates the distribution of GTDB genomes by taxon,
        then remove taxa to make the distributions more similar.

        It is recommended you first reduce the dataset to one genome per species.

        Fix: ideally, if no proportion to keep is provided, an algorithm could be used to
        determine the optimum portion to keep.

        Args:
            subset_genomes: genomes to use.
            proportion_to_keep: the fraction of genomes to keep.
            diversity_rank: what rank to use to measure diversity
        """
        balanced_genomes = set()

        # Ratio of obversed counts in data to expectation based on reference
        obs_exp_ratio_by_rank = {}
        for rank, i in self.taxonomy.indices.items():
            obs_exp_ratio_by_rank[i] = {}
            n_expected = self.taxonomy.measure_diversity(rank, diversity_rank)
            n_observed = self.taxonomy.measure_diversity(
                rank, diversity_rank, subset_genomes=genomes
            )
            for taxon, n_obs in n_observed.items():
                obs_exp_ratio = n_obs / n_expected[taxon]
                if rank == "phylum":
                    # hacky correction to keep phyla with few isolates but also few genomes
                    _reweight_rare_phyla = lambda n: (n / 500) ** 4
                    obs_exp_ratio = min([obs_exp_ratio, _reweight_rare_phyla(n_obs)])
                obs_exp_ratio_by_rank[i][taxon] = obs_exp_ratio

        # Probability of selection - should be inversely proportional
        # to degree of enrichment in observations
        probabilities = []
        for genome in genomes:
            taxonomy = self.taxonomy.taxonomy_dict.get(genome)
            if taxonomy:
                # Multiply the observation frequency over all taxonomy ranks
                obs_exp_ratio = 1.0
                for i, taxon in enumerate(taxonomy):
                    obs_exp_ratio = obs_exp_ratio * obs_exp_ratio_by_rank[i][taxon]
            else:
                # Not present in GTDB and should be removed
                obs_exp_ratio = 1.0
            p = 1 / obs_exp_ratio
            probabilities.append(p)

        # Use probability to select a certain number of genomes
        n_selections = int(proportion_to_keep * len(genomes))
        balanced_genomes = rng.choice(
            genomes,
            n_selections,
            p=probabilities / np.sum(probabilities),
            replace=False,
        )

        return set(balanced_genomes)

    def select_genomes_at_rank(
        self,
        genomes: list,
        rank: str,
        n_genomes: int = 1,
    ):
        """Selects a number of genomes for each taxon at a specified rank.

        Most useful for dereplicating species. The genome accessions are
        selected in lexigographic order. To select one genome per species:
        ```
        representative_genomes = self.select_genomes_at_rank(
                genomes=self.genomes,
                n_genomes=1,
                rank="species",
            )
        ```

        To-do: for higher taxonomic levels and n_genomes > 1, it would be more
        representative to select genomes in a more representative way.

        Args:
            genomes: list of genomes to query over
            rank: taxonomic rank to group on (e.g. species)
            n_genomes: max number of genomes to select at each taxonomic rank
        Returns:
            selected_genomes: list of genomes selected
        """

        # Group genomes by taxonomy down to the specific rank
        taxonomic_groups = defaultdict(list)
        rank_index = self.taxonomy.indices[rank]
        for genome in genomes:
            taxonomy = self.taxonomy.taxonomy_dict.get(genome, None)
            if taxonomy:
                taxonomic_groups[taxonomy[: (1 + rank_index)]].append(genome)

        # Select genomes from each group up to n_genomes
        selected_genomes = []
        for _, genome_group in taxonomic_groups.items():
            selected_genomes.extend(
                sorted(genome_group)[: min([n_genomes, len(genome_group)])]
            )

        return selected_genomes

    def assess_proportion(
        self, subset_genomes, reference_genomes, rank: str = "phylum"
    ):
        """Helper to provide composition of genomes to a reference set at specified rank."""
        n_selected = self.taxonomy.measure_diversity(
            rank, "species", subset_genomes=subset_genomes
        )
        n_reference = self.taxonomy.measure_diversity(
            rank, "species", subset_genomes=reference_genomes
        )
        return {
            taxon: n_selected.get(taxon, 0) / count
            for taxon, count in n_reference.items()
        }


class PartitionTaxa:
    """
    A tool to partition datasets using taxonomy, e.g. for making holdout sets.

    Partitioning: The user specifies the proportion of the dataset to be used
    and which taxonomic level the partition should be performed at.
    All members of a partitioned taxon (e.g. "class") will be included
    in the partition - none will not be partitioned. Partitioning is
    done randomly yet reproducibly.

    Typical usage:
        ```
        from taxonomy import PartitionTaxa, TaxonomyGTDB

        taxonomy = TaxonomyGTDB()
        partitioner = PartitionTaxa(
            taxonomy=taxonomy,
            partition_rank='family',
            iteration_rank='phylum',
            diversity_rank='genus',
        )
        partitioned_genomes = partitioner.partition(balanced_genomes, partition_size=0.2)
        nonpartitioned_genomes = set(balanced_genomes).difference(partitioned_genomes)
        extended_partitioned_genomes = partitioner.find_relatives_of_partitioned_set_in_reference(partitioned_genomes)
        ```
    Args:
        partition_rank: taxonomic rank at which to separate taxa into partitions, e.g. "family",
        iteration_rank: taxonomic rank at which to iteratively sample over, to make sure
            partitions are sourced from diverse taxa, e.g. phyla
        diversity_rank: taxonomic rank at which to compute diversity, e.g. "genus"
    """

    def __init__(
        self,
        taxonomy: TaxonomyGTDB,
        partition_rank: str = "family",
        iteration_rank: str = "phylum",
        diversity_rank: str = "genus",
    ):
        self.partition_rank = partition_rank
        self.iteration_rank = iteration_rank
        self.diversity_rank = diversity_rank
        self.taxonomy = taxonomy

    def partition(self, genomes, partition_size: float) -> set:
        """Partitions a dataset.

        To ensure a partition set is very diverse, e.g. not all from one phylum,
        a taxon is chosen by draw of one phylum at time (phylum if the
        `iteration_rank` is set to phylum). Taxa are chosen iteratively until
        the desired size of the partition is reached. Each draw occurs at a probability
        proportional to the diversity of that phylum, so that if a phylum is
        very diverse, more families can be chosen from it.

        Args:
            genomes: list of genomes to act upon
            partition_size: what portion of genomes to include in partition.
        """
        partitioned_genomes = []

        # Create dictionaries for partitioning
        partition_taxa_genomes = defaultdict(list)  # {taxon : genomes}
        partition_taxa_available = defaultdict(set)  # {taxon : randomized taxa}
        partition_tax_idx = self.taxonomy.indices[self.partition_rank]
        iteration_tax_idx = self.taxonomy.indices[self.iteration_rank]
        for genome in genomes:
            taxonomy = self.taxonomy.taxonomy_dict.get(genome, None)
            if taxonomy:
                partition_taxon = taxonomy[partition_tax_idx]
                partition_taxa_genomes[partition_taxon].append(genome)
                iterator_taxon = taxonomy[iteration_tax_idx]
                partition_taxa_available[iterator_taxon].add(partition_taxon)

        # Shuffle
        for taxon, taxa_available in partition_taxa_available.items():
            partition_taxa_available[taxon] = list(
                rng.choice(
                    list(taxa_available), size=len(taxa_available), replace=False
                )
            )
            # rng.shuffle(partition_taxa_available[taxon])

        # Set order of selections based on frequencies in dataset
        n_partition_taxa = len(
            [taxon for taxa in partition_taxa_available.values() for taxon in taxa]
        )
        iterator_counts = self.taxonomy.measure_diversity(
            self.iteration_rank, self.diversity_rank, subset_genomes=genomes
        )
        iterator_list = list(iterator_counts.keys())
        iterator_freqs = list(
            iterator_counts[phylum] / sum(iterator_counts.values())
            for phylum in iterator_list
        )
        random_iterator_order = rng.choice(
            iterator_list, n_partition_taxa, p=iterator_freqs, replace=True
        )

        # Build partition until desired size is reached
        for iteration_taxon in random_iterator_order:
            # Select available taxa e.g. families and shuffle
            available_taxa = list(partition_taxa_available.get(iteration_taxon, []))
            if len(available_taxa) > 1:
                # Add to genomes to partition and remove taxon from options
                taxon_for_partition = available_taxa[0]
                partitioned_genomes.extend(partition_taxa_genomes[taxon_for_partition])
                partition_taxa_available[iteration_taxon].remove(taxon_for_partition)
            if len(partitioned_genomes) / len(genomes) >= partition_size:
                break

        return set(partitioned_genomes)

    def find_relatives_of_partitioned_set_in_reference(
        self, partitioned_genomes: set
    ) -> set:
        """Provides all genomes within taxa selected for partitioning.

        For example, if the partition rank is family, if a genome partitioned
        from the dataset was from family A, this functions returns all genomes from family A
        in the reference set.

        This is useful for two reasons. First, balancing the dataset will remove some
        genomes from the partition, and including them in the non-partitioned data
        will represent leakage. Second, the GTDB may be updated with more genomes,
        and it will be helpful to identify genomes related to the test set and remove them.

        Args:
            partitioned_genomes: Set of genomes selected for the partition

        Returns:
            related_genomes: Set of genomes within the same taxon as partitioned genomes
        """
        taxa = self.taxonomy.taxa_of_genomes(partitioned_genomes, self.partition_rank)
        related_genomes = self.taxonomy.genomes_in_taxa(taxa, self.partition_rank)
        return related_genomes