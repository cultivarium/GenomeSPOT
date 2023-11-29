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
    def __init__(
        self,
        genomes: list,
        taxonomy_file: str,
        taxonomy_format: str = "gtdb",
        partition_size: float = 0.25,
        partition_rank: str = "family",
        iteration_rank: str = "phylum",
        diversity_rank: str = "genus",
        exclude_genomes: list = [],
    ):
        self.genomes = genomes
        self.partition_size = partition_size
        self.partition_rank = partition_rank
        self.iteration_rank = iteration_rank
        self.diversity_rank = diversity_rank
        self.exclude_genomes = exclude_genomes

        self.taxonomy_indices = {}
        self.taxonomy_dict = self.make_taxonomy_dict(
            filename=taxonomy_file, source=taxonomy_format
        )

    def make_taxonomy_dict(self, filename: str, source: str = "gtdb") -> dict:
        """
        Load a taxonomic file assigning a genome to a taxonomy

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

        if source == "gtdb":
            self.taxonomy_indices = {
                "domain": 0,
                "phylum": 1,
                "class": 2,
                "order": 3,
                "family": 4,
                "genus": 5,
                "species": 6,
            }

            with open(filename) as fh:
                for line in fh.readlines():
                    gtdb_accession, taxstring = line.strip().split("\t")
                    ncbi_accession = gtdb_accession_to_ncbi(gtdb_accession)
                    _, taxonomy = gtdb_taxonomy_to_tuple(taxstring)
                    taxonomy_dict[ncbi_accession] = taxonomy

        else:
            raise ValueError("Only supported `source` is `gtdb`")

        return taxonomy_dict

    def measure_diversity(
        self, query_rank: str, diversity_rank: str, subset_genomes: list = None
    ) -> dict:
        """Measure diversity of each taxon.

        Counts the number of taxa at rank `diversity_rank`
        under each taxon of the `query_rank`.
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

    def select_genomes_at_rank(
        self,
        genomes: list,
        rank: str,
        n_genomes: int = 1,
    ):
        """Selects a number of genomes for each taxon at a specified rank.

        The genome accessions are selected in lexigographic order. To select
        one genome per species, set rank='species' and n_genomes=1.

        Fix: for higher taxonomic levels and n_genomes > 1, it would be more
        representative to selected genomes for in a more representative way.

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
                    # Correction to keep phyla with few isolates but also few genomes
                    _reweight_rare_clades = lambda n: (n / 500) ** 4
                    obs_exp_ratio = min([obs_exp_ratio, _reweight_rare_clades(n_obs)])
                obs_exp_ratio_by_rank[i][taxon] = obs_exp_ratio

        # Probability of selection should be inversely proportional
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

        # Select a certain number of genomes based on this proportion
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
        partition_taxa_genomes = defaultdict(list)
        partition_taxa_available = defaultdict(set)
        for genome in subset_genomes:
            taxonomy = self.taxonomy_dict.get(genome, None)
            if taxonomy:
                partition_taxon = taxonomy[self.taxonomy_indices[self.partition_rank]]
                iterator_taxon = taxonomy[self.taxonomy_indices[self.iteration_rank]]
                # Add to dictionary {partition taxon : list(genomes)}
                partition_taxa_genomes[partition_taxon].append(genome)
                # Add to dictionary {iterator taxon : random set(partition taxa)}
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
            options = list(partition_taxa_available.get(iteration_taxon, []))
            if len(options) > 1:
                # Add to genomes to partition and remove taxon from options
                taxon_for_partition = options[0]
                partitioned_genomes.extend(partition_taxa_genomes[taxon_for_partition])
                partition_taxa_available[iteration_taxon].remove(taxon_for_partition)
            if len(partitioned_genomes) / len(subset_genomes) >= partition_size:
                break

        return set(partitioned_genomes)

    def balanced_partition(
        self,
        proportion_to_keep: float = None,
        dereplicate_species: bool = True,
    ) -> dict:
        """Workflow to run balancing and partitioning"""

        if dereplicate_species:
            # Reduce to one genome per species
            representative_genomes = self.select_genomes_at_rank(
                genomes=self.genomes,
                n_genomes=1,
                rank="species",
            )
        else:
            representative_genomes = self.genomes

        balanced_genomes = self.balance_dataset(
            dereplicate_species=dereplicate_species,
            calibrate_to_phylogeny=True,
            proportion_to_keep=proportion_to_keep,
        )

        # Partition
        partitioned_genomes = set(self.partition(subset_genomes=balanced_genomes))
        retained_genomes = set(balanced_genomes).difference(set(partitioned_genomes))

        return {"in": retained_genomes, "out": partitioned_genomes}

    def get_genomes_in_partitioned_taxa(self, partitioned_genomes: set) -> set:
        """Provides all genomes within taxa selected for partitioning.

        For example, if the partition rank is family, if a partitioned
        genome was from family A, all genomes from family A would be returned.
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

    def assess_distance_to_reference(self, subset_genomes, rank: str = "phylum"):
        """Bray-Curtis dissimilarity of composition to taxonomy reference"""
        n_reference = self.measure_diversity(rank, "species")
        n_maximum = self.measure_diversity(rank, "species", subset_genomes=self.genomes)
        n_selected = self.measure_diversity(
            rank, "species", subset_genomes=subset_genomes
        )
        arr_selected = [n_selected.get(family, 0) for family in n_maximum.keys()]
        arr_expected = [n_reference.get(family, 0) for family in n_maximum.keys()]
        abundance_selected = arr_selected / np.sum(arr_selected)
        abundance_expected = arr_expected / np.sum(arr_expected)
        return distance.braycurtis(abundance_expected, abundance_selected)

    def assess_proportion(
        self, subset_genomes, reference_genomes, rank: str = "phylum"
    ):
        """Compositional fraction of genomes to a reference set at specified rank"""
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
