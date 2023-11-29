"""
Uses phylogeny to partition a dataset.
"""

from collections import defaultdict, Counter

import numpy as np
from scipy.spatial import distance

from helpers import gtdb_accession_to_ncbi, gtdb_taxonomy_to_tuple

RANDOM_SEED = 1111
rng = np.random.default_rng(RANDOM_SEED)


class PhylogeneticPartition:
    """
    A tool to balance and partition datasets using taxonomy.

    Partitioning: The user specifies the proportion of the dataset to be used
    and which taxonomic level the partition should be performed at.
    All members of a partitioned taxon (e.g. "class") will be included
    in the partition - none will not be partitioned. Partitioning is
    done randomly yet reproducibly.

    Balancing: The user specifies a number of taxa to keep, and taxa are
    randomly chosen with a higher likelihood of being chosen inversely
    proportional to the bias for that taxon in the dataset compared to the
    supplied taxonomic reference.

    Args:
        genomes: list of genomes to act upon
        taxonomy_filenames: list of taxonomy files,
        taxonomy_format: format of taxonomy file (only 'gtdb' allowed currently)
        partition_size: portion of the dataset to yield in partition, e.g. 0.25
        partition_rank: taxonomic rank at which to separate taxa into partitions, e.g. "family",
        iteration_rank: taxonomic rank at which to iteratively sample over, to make sure
            partitions are sourced from diverse taxa, e.g. phyla
        diversity_rank: taxonomic rank at which to compute diversity, e.g. "genus"
    """

    def __init__(
        self,
        # genomes: list,
        taxonomy_filenames: list,
        taxonomy_format: str = "gtdb",
        partition_size: float = 0.25,
        partition_rank: str = "family",
        iteration_rank: str = "phylum",
        diversity_rank: str = "genus",
        # exclude_genomes: list = [],
    ):
        # self.genomes = genomes
        self.partition_size = partition_size
        self.partition_rank = partition_rank
        self.iteration_rank = iteration_rank
        self.diversity_rank = diversity_rank
        # self.exclude_genomes = exclude_genomes
        self.taxonomy_filenames = taxonomy_filenames
        self.taxonomy_indices = {}
        self.taxonomy_format = taxonomy_format
        self.taxonomy_dict = self.make_taxonomy_dict()

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

        Args:

        """

        taxonomy_dict = {}

        if self.taxonomy_format == "gtdb":
            self.taxonomy_indices = {
                "domain": 0,
                "phylum": 1,
                "class": 2,
                "order": 3,
                "family": 4,
                "genus": 5,
                "species": 6,
            }
            for filename in self.taxonomy_filenames:
                with open(filename, "r") as fh:
                    for line in fh.readlines():
                        gtdb_accession, taxstring = line.strip().split("\t")
                        ncbi_accession = gtdb_accession_to_ncbi(gtdb_accession)
                        _, taxonomy = gtdb_taxonomy_to_tuple(taxstring)
                        taxonomy_dict[ncbi_accession] = taxonomy

        else:
            raise ValueError("Only supported `taxonomy_format` is `gtdb`")

        return taxonomy_dict

    def measure_diversity(
        self, query_rank: str, diversity_rank: str, subset_genomes: list = None
    ) -> dict:
        """Counts the number of taxa at rank `diversity_rank`
        under each taxon of the `query_rank` for a set of genomes.
        """
        query_index = self.taxonomy_indices[query_rank]
        diversity_index = self.taxonomy_indices[diversity_rank]
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

    def balance_dataset(
        self,
        subset_genomes: list,
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
        for rank, i in self.taxonomy_indices.items():
            obs_exp_ratio_by_rank[i] = {}
            n_expected = self.measure_diversity(rank, diversity_rank)
            n_observed = self.measure_diversity(
                rank, diversity_rank, subset_genomes=subset_genomes
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
        for genome in subset_genomes:
            taxonomy = self.taxonomy_dict.get(genome)
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
        n_selections = int(proportion_to_keep * len(subset_genomes))
        balanced_genomes = rng.choice(
            subset_genomes,
            n_selections,
            p=probabilities / np.sum(probabilities),
            replace=False,
        )

        return set(balanced_genomes)

    def partition(self, subset_genomes: set, partition_size: float) -> set:
        """Partitions a dataset.

        Ensure holdouts are from across diverse phyla by iteratively selecting
        a phylum to sample a family from with the probability of choosing a
        phylum equal to the diversity of that phylum. Continue to sample families
        until desired proportion of data is reached.

        Args:
            subset_genomes: list of genomes to create partition from.
            partition_size: what portion of genomes to include in partition.
        """
        partitioned_genomes = []

        # Create dictionaries for partitioning
        partition_taxa_genomes = defaultdict(list)  # {taxon : genomes}
        partition_taxa_available = defaultdict(set)  # {taxon : randomized taxa}
        for genome in subset_genomes:
            taxonomy = self.taxonomy_dict.get(genome, None)
            if taxonomy:
                partition_taxon = taxonomy[self.taxonomy_indices[self.partition_rank]]
                partition_taxa_genomes[partition_taxon].append(genome)
                iterator_taxon = taxonomy[self.taxonomy_indices[self.iteration_rank]]
                partition_taxa_available[iterator_taxon].add(partition_taxon)

        # Shuffle
        for taxon, taxa_available in partition_taxa_available.items():
            partition_taxa_available[taxon] = list(taxa_available)
            rng.shuffle(partition_taxa_available[taxon])

        # Set order of selections based on frequencies in dataset
        n_partition_taxa = len(
            [taxon for taxa in partition_taxa_available.values() for taxon in taxa]
        )
        iterator_counts = self.measure_diversity(
            self.iteration_rank, self.diversity_rank, subset_genomes=subset_genomes
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
            if len(partitioned_genomes) / len(subset_genomes) >= partition_size:
                break

        return set(partitioned_genomes)

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
        rank_index = self.taxonomy_indices[rank]
        for genome in genomes:
            taxonomy = self.taxonomy_dict.get(genome, None)
            if taxonomy:
                taxonomic_groups[taxonomy[: (1 + rank_index)]].append(genome)

        # Select genomes from each group up to n_genomes
        selected_genomes = []
        for _, genome_group in taxonomic_groups.items():
            selected_genomes.extend(
                sorted(genome_group)[: min([n_genomes, len(genome_group)])]
            )

        return selected_genomes

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

        partitioned_taxa = []
        for genome in partitioned_genomes:
            taxonomy = self.taxonomy_dict.get(genome, None)
            if taxonomy:
                partitioned_taxa.append(
                    taxonomy[self.taxonomy_indices[self.partition_rank]]
                )

        related_genomes = []
        for genome, taxonomy in self.taxonomy_dict.items():
            taxon = taxonomy[self.taxonomy_indices[self.partition_rank]]
            if taxon in partitioned_taxa:
                related_genomes.append(genome)

        return set(related_genomes)

    def assess_proportion(
        self, subset_genomes, reference_genomes, rank: str = "phylum"
    ):
        """Helper to provide composition of genomes to a reference set at specified rank."""
        n_selected = self.measure_diversity(
            rank, "species", subset_genomes=subset_genomes
        )
        n_reference = self.measure_diversity(
            rank, "species", subset_genomes=reference_genomes
        )
        return {
            taxon: n_selected.get(taxon, 0) / count
            for taxon, count in n_reference.items()
        }
