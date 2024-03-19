"""BalanceTaxa: balances a dataset using taxonomy"""

from collections import defaultdict

import numpy as np

from .taxonomy import TaxonomyGTDB


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
        proportion_to_keep: float,
        diversity_rank: str = "species",
    ):
        """
        Balance taxonomic groups by removing overrepresented taxa:
        observe the distribution of available
        genomes by taxon relates the distribution of GTDB species by taxon,
        then remove taxa to make the distributions more similar.

        To-do: ideally, if no proportion to keep is provided, an algorithm could be used to
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
            n_observed = self.taxonomy.measure_diversity(rank, diversity_rank, subset_genomes=genomes)
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
        rng = np.random.default_rng(seed=12345)
        balanced_genomes = rng.choice(
            genomes,
            n_selections,
            p=probabilities / np.sum(probabilities),
            replace=False,
        )

        return sorted(set(balanced_genomes))

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
            selected_genomes.extend(sorted(genome_group)[: min([n_genomes, len(genome_group)])])

        return selected_genomes

    def assess_proportion(self, subset_genomes, reference_genomes, rank: str = "phylum"):
        """Helper to provide composition of genomes to a reference set at specified rank."""
        n_selected = self.taxonomy.measure_diversity(rank, "species", subset_genomes=subset_genomes)
        n_reference = self.taxonomy.measure_diversity(rank, "species", subset_genomes=reference_genomes)
        return {taxon: n_selected.get(taxon, 0) / count for taxon, count in n_reference.items()}
