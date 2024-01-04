import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..helpers import (
    load_train_and_test_sets,
    rename_condition_to_variable,
)
from ..taxonomy import (
    BalanceTaxa,
    PartitionTaxa,
    TaxonomyGTDB,
)


ROOT_DIR = str(Path(__file__).resolve().parent.parent)

HOLDOUT_DIRECTORY = f"{ROOT_DIR}/data/holdouts/"
TRAINING_DATA_TSV = f"{ROOT_DIR}/data/training_data/training_data_20231117.tsv"

THRESHOLDS_TO_KEEP = {
    "oxygen": (0, 1),  # i.e. no thresholds
    "salinity": (0, 14),  # i.e. no lower threshold
    "temperature": (19, 45),
    "ph": (4, 9),
}

BALANCE_PROPORTIONS = {
    "oxygen": 0.5,
    "salinity": 2 / 3,
    "temperature": 2 / 3,
    "ph": 2 / 3,
}


def balance_but_keep_extremes(
    df_data,
    target,
    genomes_for_use,
    balancer,
):
    # Remove genomes by bias in dataset
    condition = target.split("_")[0]
    balance_proportion = BALANCE_PROPORTIONS[condition]
    balanced_genomes = balancer.balance_dataset(
        genomes=genomes_for_use, proportion_to_keep=balance_proportion, diversity_rank="species"
    )

    logging.info(
        "%s: Genomes remaining after balancing: %i (frac==%.2f)",
        target,
        len(balanced_genomes),
        len(balanced_genomes) / len(genomes_for_use),
    )
    # Return genomes from extremes:
    keep_below, keep_above = THRESHOLDS_TO_KEEP[condition]
    return_high_genomes = set(df_data.loc[df_data[target] >= keep_above].index.tolist())
    return_low_genomes = set(df_data.loc[df_data[target] <= keep_below].index.tolist())
    return_genomes = return_high_genomes.union(return_low_genomes)
    balanced_genomes = balanced_genomes.union(return_genomes)
    logging.info(
        "%s: Genomes after %i extremophiles returned: %i (frac==%.2f)",
        target,
        len(return_genomes),
        len(balanced_genomes),
        len(balanced_genomes) / len(genomes_for_use),
    )
    return sorted(balanced_genomes)


def make_balanced_partitions_for_variable(
    df_data,
    target,
    balancer,
    partition_size,
    partitioner,
    percentile_bins,
):
    genomes_for_use = sorted(df_data.index.drop_duplicates().tolist())
    logging.info(
        "%s: Genomes input for use: %i",
        target,
        len(genomes_for_use),
    )

    balanced_genomes = balance_but_keep_extremes(
        df_data,
        target,
        genomes_for_use,
        balancer,
    )

    balanced_df = df_data.loc[list(balanced_genomes)]
    partitioned_genomes = partition_within_percentiles(
        balanced_df, target, percentile_bins, partitioner, partition_size
    )
    extended_partitioned_genomes = partitioner.find_relatives_of_partitioned_set_in_reference(partitioned_genomes)

    test_set = partitioned_genomes
    train_set = set(balanced_genomes).difference(test_set)
    logging.info(
        "%s: Genomes in test and train partitions: %i (frac=%.2f) / %i (frac=%.2f)",
        target,
        len(test_set),
        len(test_set) / len(balanced_genomes),
        len(train_set),
        len(train_set) / len(balanced_genomes),
    )

    return sorted(test_set), sorted(train_set)


def partition_within_percentiles(balanced_df, target, percentile_bins, partitioner, partition_size):
    """Performs a partition within each percentile range of data

    Most useful for including extreme percentiles in data, e.g. <5% and >95%.
    """
    partitioned_genomes = set()
    for percentile_min, percentile_max in percentile_bins:
        bin_min = np.percentile(balanced_df[target].values, percentile_min)
        bin_max = np.percentile(balanced_df[target].values, percentile_max)
        if percentile_max == 100:
            balanced_genomes_in_bin = balanced_df[balanced_df[target] >= bin_min].index.tolist()
        else:
            balanced_genomes_in_bin = balanced_df[
                (balanced_df[target] >= bin_min) & (balanced_df[target] < bin_max)
            ].index.tolist()
        partitioned_genomes_in_bin = partitioner.partition(balanced_genomes_in_bin, partition_size=partition_size)
        partitioned_genomes = partitioned_genomes.union(partitioned_genomes_in_bin)
        logging.info(
            "%s: Genomes partitioned in percentiles %i-%i: %i",
            target,
            percentile_min,
            percentile_max,
            len(partitioned_genomes_in_bin),
        )
    return sorted(partitioned_genomes)


def save_partitions(test_set, train_set, test_file, train_file, overwrite=True):
    if Path(test_file).exists() and overwrite == False:
        pass
    else:
        with open(test_file, "w") as fh:
            fh.write("\n".join(test_set))
        with open(train_file, "w") as fh:
            fh.write("\n".join(train_set))


def make_holdout_sets(
    df,
    partition_size=0.20,
    balance_proportion=2 / 3,
    output_directory=HOLDOUT_DIRECTORY,
    overwrite=True,
):
    taxonomy = TaxonomyGTDB()
    balancer = BalanceTaxa(taxonomy=taxonomy)
    partitioner = PartitionTaxa(
        taxonomy=taxonomy,
        partition_rank="family",
        iteration_rank="phylum",
        diversity_rank="genus",
    )

    holdout_sets = {}
    for condition in ["oxygen", "salinity", "ph", "temperature"]:
        test_file = f"{output_directory}/test_set_{condition}.txt"
        train_file = f"{output_directory}/train_set_{condition}.txt"
        if condition == "oxygen":
            target = condition
            percentile_bins = [(0, 100)]
        else:
            target = condition + "_optimum"
            percentile_bins = [(0, 3), (3, 97), (97, 100)]

        df_data = df[(df[f"use_{condition}"] == True)]
        test_set, train_set = make_balanced_partitions_for_variable(
            df_data,
            target,
            balancer,
            partition_size,
            partitioner,
            percentile_bins,
        )

        save_partitions(test_set, train_set, test_file, train_file, overwrite=overwrite)

        holdout_sets[target] = {"test": test_set, "train": train_set}

    return holdout_sets


def make_cv_sets_by_phylogeny(genomes: np.ndarray, partition_rank="family", kfold=5):
    partitioner = PartitionTaxa(
        taxonomy=TaxonomyGTDB(),
        partition_rank=partition_rank,
        iteration_rank="phylum",
        diversity_rank="species",
    )

    cv_sets = []  # training, validation
    remaining_genomes = set(genomes)
    fold_sizes = []
    for k in range(kfold):
        validation_partition_size = 1 / (kfold - k)
        partitioned_genomes = partitioner.partition(remaining_genomes, partition_size=validation_partition_size)
        if k < kfold - 1:
            remaining_genomes = set(remaining_genomes).difference(partitioned_genomes)
        else:
            partitioned_genomes = remaining_genomes
        validation_indices = np.in1d(genomes, np.array(list(partitioned_genomes))).nonzero()[0]
        training_indices = np.in1d(genomes, np.array(list(set(genomes).difference(partitioned_genomes)))).nonzero()[0]
        cv_sets.append((training_indices, validation_indices))
        fold_sizes.append(len(partitioned_genomes) / len(genomes))
        # print( len(partitioned_genomes), len(remaining_genomes),  validation_partition_size, len(partitioned_genomes) / len(genomes), )
    logging.info("Fold sizes for k=%i: %s", kfold, ", ".join([f"{f:.2f}" for f in fold_sizes]))
    # Check
    n_valid = 0
    for indices_train, indices_valid in cv_sets:
        n_valid += len(indices_valid)
    assert n_valid == len(genomes)
    return cv_sets


def make_cv_sets_randomly(genomes: np.ndarray, partition_rank="family", kfold=5):
    rng = np.random.default_rng(seed=0)
    indices = np.array([i for i in range(len(genomes))])
    randomized_indices = rng.choice(indices, size=len(genomes), replace=False)
    split = int(len(genomes) / kfold)
    cv_sets = []
    for k in range(kfold):
        if k < kfold - 1:
            validation_indices = randomized_indices[k * split : (k + 1) * split]
        else:
            validation_indices = randomized_indices[k * split :]
        training_indices = np.array(list(set(randomized_indices).difference(set(validation_indices))))
        cv_sets.append((training_indices, validation_indices))
    return cv_sets


def convert_cv_sets_to_json_format(cv_sets):
    json_serializable_sets = []
    for training_indices, validation_indices in cv_sets:
        serializable_set = ([int(idx) for idx in training_indices], [int(idx) for idx in validation_indices])
        json_serializable_sets.append(serializable_set)
    return json_serializable_sets


def make_cv_sets_for_model_evaluation(df, path_to_holdouts, overwrite=True):
    for condition in ["oxygen", "temperature", "ph", "salinity"]:
        cv_sets_dict = {}
        target = rename_condition_to_variable(condition)
        train_set, _ = load_train_and_test_sets(condition, path_to_holdouts)
        df_train = df.loc[train_set]
        for partition_rank in ["phylum", "class", "order", "family", "genus", "species"]:
            logging.info("Generating CV sets for %s at level %s", condition, partition_rank)
            cv_sets = make_cv_sets_by_phylogeny(genomes=df_train.index.tolist(), partition_rank=partition_rank, kfold=5)
            print(len(cv_sets))
            cv_sets_dict[partition_rank] = convert_cv_sets_to_json_format(cv_sets)

        if overwrite is True:
            json.dump(cv_sets_dict, open(f"{path_to_holdouts}/{condition}_cv_sets.json", "w"))


if __name__ == "__main__":
    make_holdout_sets(df=pd.read_csv(TRAINING_DATA_TSV), overwrite=True)
