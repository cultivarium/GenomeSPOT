# pylint: disable=missing-docstring
from pathlib import Path

from genome_spot.taxonomy.balance import BalanceTaxa
from genome_spot.taxonomy.taxonomy import TaxonomyGTDB


cwd = Path(__file__).resolve().parent
TAXONOMY_FILENAMES = [f"{cwd}/test_data/test_ar53_taxonomy_r214.tsv", f"{cwd}/test_data/test_bac120_taxonomy_r214.tsv"]


class TestBalanceTaxa:

    def test_balance_dataset(self):

        expected_values = ["GCA_003216535", "GCA_003513495", "GCA_005801235", "GCA_006348925"]
        with open(f"{cwd}/test_data/test_genome_accessions.txt", "r") as fh:
            genomes = [line.strip() for line in fh.readlines()]
        taxonomy = TaxonomyGTDB(TAXONOMY_FILENAMES)
        balancer = BalanceTaxa(taxonomy=taxonomy)
        balanced_genomes = balancer.balance_dataset(genomes=genomes, proportion_to_keep=0.02, diversity_rank="species")

        assert balanced_genomes == expected_values
