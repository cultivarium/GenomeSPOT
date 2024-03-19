# pylint: disable=missing-docstring
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from genome_spot.model_training.make_holdout_sets import (
    make_cv_sets_by_phylogeny,
    partition_within_percentiles,
)
from genome_spot.taxonomy.partition import PartitionTaxa
from genome_spot.taxonomy.taxonomy import TaxonomyGTDB


cwd = Path(__file__).resolve().parent
TAXONOMY_FILENAMES = [f"{cwd}/test_data/test_ar53_taxonomy_r214.tsv", f"{cwd}/test_data/test_bac120_taxonomy_r214.tsv"]

with open(f"{cwd}/test_data/test_genome_accessions.txt", "r") as fh:
    GENOMES = [line.strip() for line in fh.readlines()]


def test_partition_within_percentiles():

    # Create mock dataframe
    mock_df = pd.DataFrame(
        {"ncbi_accession": GENOMES, "oxygen": np.random.choice([1, 0], len(GENOMES)), "use_oxygen": True}
    ).set_index("ncbi_accession")

    taxonomy = TaxonomyGTDB(TAXONOMY_FILENAMES)
    partitioner = PartitionTaxa(taxonomy, partition_rank="family", diversity_rank="genus")
    partitioned_genomes = partition_within_percentiles(
        balanced_df=mock_df,
        target="oxygen",
        percentile_bins=[(0, 100)],
        partitioner=partitioner,
        partition_size=0.50,
    )

    # Sampling percentile bounds correctly
    assert len(partitioned_genomes) == 100

    # No overlap between families
    partitioned_families = set()
    for genome in partitioned_genomes:
        partitioned_families.add(taxonomy.taxonomy_dict[genome][4])
    non_partitioned_families = set()
    for genome in set(GENOMES).difference(set(partitioned_genomes)):
        non_partitioned_families.add(taxonomy.taxonomy_dict[genome][4])
    n_overlap = len(partitioned_families.intersection(non_partitioned_families))
    assert n_overlap == 0


def test_make_cv_sets_by_phylogeny():

    k = 5
    cv_sets = make_cv_sets_by_phylogeny(GENOMES, partition_rank="family", kfold=k)

    # Check CV sets are non-overlapping and complete
    all_validation_indices = []
    for training_indices, validation_indices in cv_sets:
        all_validation_indices.extend(list(validation_indices))
        assert len(list(training_indices) + list(validation_indices)) == len(GENOMES)
        assert len(set(training_indices).intersection(set(validation_indices))) == 0
        assert max(Counter(all_validation_indices).values()) == 1  # no duplicates
    assert len(cv_sets) == k
    assert len(all_validation_indices) == len(GENOMES)
