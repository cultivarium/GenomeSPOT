# pylint: disable=missing-docstring
from pathlib import Path

from genome_spot.taxonomy.partition import PartitionTaxa
from genome_spot.taxonomy.taxonomy import TaxonomyGTDB


cwd = Path(__file__).resolve().parent
TAXONOMY_FILENAMES = [f"{cwd}/test_data/test_ar53_taxonomy_r214.tsv", f"{cwd}/test_data/test_bac120_taxonomy_r214.tsv"]


class TestPartitionTaxa:

    def test_partition(self):

        with open(f"{cwd}/test_data/test_genome_accessions.txt", "r") as fh:
            genomes = [line.strip() for line in fh.readlines()]

        taxonomy = TaxonomyGTDB(TAXONOMY_FILENAMES)
        partitioner = PartitionTaxa(
            taxonomy=taxonomy,
            partition_rank="family",
            diversity_rank="genus",
        )
        partitioned_genomes = partitioner.partition(genomes, partition_size=0.02)
        expected_values = {"GCA_016190825", "GCA_020056635", "GCA_900549505", "GCA_903888175"}
        assert partitioned_genomes == expected_values
