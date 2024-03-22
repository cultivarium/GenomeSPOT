"""PartitionTax partitions a dataset using taxonomy"""

from collections import defaultdict

import numpy as np

from .taxonomy import TaxonomyGTDB


class PartitionTaxa:
    """
    A tool to partition datasets using taxonomy, e.g. for making holdout sets.

    Partitioning: The user specifies the proportion of the dataset to be used
    and which taxonomic level the partition should be performed at.
    All members of a partitioned taxon (e.g. "class") will be included
    in the partition - none will not be partitioned. Partitioning is
    done with a reproducible random number generator. Downstream use
    of partitioning should use sorted sets instead of unordered sets to maintain
    reproducibility.

    Typical usage:
        ```
        from taxonomy import PartitionTaxa, TaxonomyGTDB

        taxonomy = TaxonomyGTDB()
        partitioner = PartitionTaxa(
            taxonomy=taxonomy,
            partition_rank='family',
            diversity_rank='genus',
        )
        partitioned_genomes = partitioner.partition(balanced_genomes, partition_size=0.2)
        nonpartitioned_genomes = set(balanced_genomes).difference(partitioned_genomes)
        extended_partitioned_genomes = partitioner.find_relatives_of_partitioned_set_in_reference(partitioned_genomes)
        ```
    Args:
        partition_rank: taxonomic rank at which to separate taxa into partitions, e.g. "family",
        diversity_rank: taxonomic rank at which to compute diversity, e.g. "genus"
    """

    def __init__(
        self,
        taxonomy: TaxonomyGTDB,
        partition_rank: str = "family",
        diversity_rank: str = "genus",
    ):
        self.partition_rank = partition_rank
        self.diversity_rank = diversity_rank
        self.taxonomy = taxonomy

    def partition(self, genomes, partition_size: float) -> set:
        partitioned_genomes = []

        # Get taxa
        taxa = list(self.taxonomy.taxa_of_genomes(genomes, self.partition_rank))
        taxon_to_genomes = defaultdict(list)
        taxlevel_idx = self.taxonomy.indices[self.partition_rank]
        for genome in genomes:
            genome_taxonomy = self.taxonomy.taxonomy_dict.get(genome, None)
            if genome_taxonomy is not None:
                taxon_to_genomes[genome_taxonomy[taxlevel_idx]].append(genome)

        # Create random order of taxa at partition_rank
        rng = np.random.default_rng(seed=12345)
        random_order = rng.choice(taxa, size=len(taxa), replace=False)

        # Add to genomes to partition until desired size is reached
        for taxon in random_order:
            partitioned_genomes.extend(taxon_to_genomes[taxon])
            if len(partitioned_genomes) / len(genomes) >= partition_size:
                break

        return set(partitioned_genomes)

    def find_relatives_of_partitioned_set_in_reference(self, partitioned_genomes: set) -> set:
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
