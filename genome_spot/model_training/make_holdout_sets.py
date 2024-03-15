"""Functions for making holdout sets for model training and evaluation

As a script, runs one set of functions to generate test/train holdout sets and cross-validation sets.

For holdout sets, the test and train sets are saved as lists of genome accessions.

For cross-validation, the cross-validation sets are a list of tuples where
the first item is an array of genome accessions of data to be used in training and the second 
item is an array of genome accessions to serve as validation data. To be used with scikit-learn
functions, these need to be provided as a generator object.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import (
    List,
    Tuple,
)

import numpy as np
import pandas as pd

from ..helpers import (
    load_train_and_test_sets,
    load_training_data,
)
from ..taxonomy.balance import BalanceTaxa
from ..taxonomy.partition import PartitionTaxa
from ..taxonomy.taxonomy import TaxonomyGTDB


logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", format="%(asctime)s %(levelname)s %(message)s")

CONDITIONS = ["oxygen", "salinity", "ph", "temperature"]

THRESHOLDS_TO_KEEP = {
    "oxygen": (-9999, 9999),  # i.e. no thresholds
    "salinity": (-1, 14),  # i.e. no lower threshold
    "temperature": (19, 45),
    "ph": (4, 9),
}

BALANCE_PROPORTIONS = {
    "oxygen": 0.5,
    "salinity": 2 / 3,
    "temperature": 2 / 3,
    "ph": 2 / 3,
}


### FUNCTIONS FOR TRAIN / TEST HOLDOUT SETS


def make_holdout_sets(
    df: pd.DataFrame,
    path_to_holdouts: str,
    partition_size: float = 0.20,
    overwrite: bool = True,
):
    """Main function to make balanced test/train partitions for all conditions

    Args:
        df: DataFrame of  data
        path_to_holdouts: Path to directory to save holdout sets
        partition_size: Fraction of genomes to be in the 'test' set
        balance_proportion: Fraction of genomes to be kept after balancing
        overwrite: If True, saves file, overwriting any existing holdout sets. If False, does not write to files.

    Returns:
        Dictionary of holdout sets, keyed by each condition, e.g. 'oxygen', 'salinity', etc.
    """
    taxonomy = TaxonomyGTDB()
    balancer = BalanceTaxa(taxonomy=taxonomy)
    partitioner = PartitionTaxa(
        taxonomy=taxonomy,
        partition_rank="family",
        diversity_rank="genus",
    )

    holdout_sets = {}
    for condition in CONDITIONS:
        logging.info("Generating holdout sets for %s", condition)
        test_file = f"{path_to_holdouts}/test_set_{condition}.txt"
        train_file = f"{path_to_holdouts}/train_set_{condition}.txt"
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


def make_balanced_partitions_for_variable(
    df_data: pd.DataFrame,
    target: str,
    balancer: BalanceTaxa,
    partition_size: float,
    partitioner: PartitionTaxa,
    percentile_bins: list,
) -> Tuple[list, list]:
    """Make balanced partitions for a variable, keeping extreme values

    For each condition, e.g. oxygen, salinity, etc.:
    - Performs a phylogenetic balancing of the entire dataset; then
    - Partitions a fraction of genomes to be in the 'test' set

    Balancing is performed by removing genomes from the dataset, but keeping genomes
    that produced a phylogenetically diverse set of genomes. The proportions kept
    are specified in module variable BALANCE_PROPORTIONS. Values beyond the thresholds
    specified in THRESHOLDS_TO_KEEP are kept.

    Partitioning occurs within percentiles of the target variable. For example,
    for salinity, the partitioning occurs independently between the 0-3rd percentile,
    3-97 percentile, and 97-100th percentile of salinity values. This ensures that
    the test set contains genomes from the extremes of the target variable.

    Args:
        df_data: DataFrame of data to balance
        target: Target variable of data being balanced
        balancer: Instance of BalanceTaxa class
        partition_size: Fraction of genomes to be in the 'test' set
        partitioner: Instance of PartitionTaxa class
        percentile_bins: List of tuples specifying the percentile ranges to partition within

    Returns:
        Tuple of lists of genome accessions in the test and train sets
    """
    genomes_for_use = sorted(df_data.index.drop_duplicates().tolist())
    logging.info(
        "%s: Genomes input for use: %i",
        target,
        len(genomes_for_use),
    )

    condition = target.split("_")[0]
    balance_proportion = BALANCE_PROPORTIONS[condition]
    keep_below, keep_above = THRESHOLDS_TO_KEEP[condition]
    balanced_genomes = balance_but_keep_extremes(
        df_data,
        target,
        genomes_for_use,
        balancer,
        balance_proportion=balance_proportion,
        keep_below=keep_below,
        keep_above=keep_above,
    )
    logging.info("%s: Genomes after balancing: %i", target, len(balanced_genomes))

    balanced_df = df_data.loc[list(balanced_genomes)]
    partitioned_genomes = partition_within_percentiles(
        balanced_df, target, percentile_bins, partitioner, partition_size
    )

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


def balance_but_keep_extremes(
    df_data: pd.DataFrame,
    target: str,
    genomes_for_use: list,
    balancer: BalanceTaxa,
    balance_proportion: float,
    keep_below: float,
    keep_above: float,
) -> list:
    """Produce a more phylogenetically balanced dataset by removing genomes,
    but keep genomes at extreme values of the target variable.

    Args:
        df_data: DataFrame of data to balance
        target: Target variable of data being balanced
        genomes_for_use: List of genomes to use for balancing
        balancer: Instance of BalanceTaxa class
    Returns:
        List of genome accessions to use for training
    """
    # Remove genomes by bias in dataset

    balanced_genomes = set(
        balancer.balance_dataset(
            genomes=genomes_for_use, proportion_to_keep=balance_proportion, diversity_rank="species"
        )
    )

    logging.info(
        "%s: Genomes remaining after balancing: %i (frac==%.2f)",
        target,
        len(balanced_genomes),
        len(balanced_genomes) / len(genomes_for_use),
    )
    # Return genomes from extremes:
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


def partition_within_percentiles(
    balanced_df: pd.DataFrame,
    target: str,
    percentile_bins: List[Tuple[float, float]],
    partitioner: PartitionTaxa,
    partition_size: float,
) -> list:
    """Performs a partition within each percentile range of data.

    Used to ensure that extreme percentiles data, e.g. <5% and >95%, are present
    in both the training data and the test data. However, as taxa in those percentiles
    are partitioned independently, this leads to data leakage at a taxonomic level.
    The impact of this data leakage on interpreting model results is limited by the fact
    that the taxa may be closely related but likely have very different traits.

    Args:
        balanced_df: DataFrame of data to partition
        target: Target variable of data being partitioned
        percentile_bins: List of tuples specifying the percentile ranges to partition within
        partitioner: Instance of PartitionTaxa class
        partition_size: Fraction of genomes to be in the 'test' set
    Return:
        List of genome accessions to use for training
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


def save_partitions(test_set: list, train_set: list, test_file: str, train_file: str, overwrite: bool = True):
    """Save test and train sets to files unless
    overwrite is False and files already exist

    Args:
        test_set: List of genome accessions to use for testing
        train_set: List of genome accessions to use for training
        test_file: Path to file to save test set
        train_file: Path to file to save train set
        overwrite: If True, saves file, overwriting any existing holdout sets. If False, does not write to files.
    Returns:
        None
    """
    if Path(test_file).exists() and overwrite == False:
        pass
    elif overwrite is True:
        with open(test_file, "w") as fh:
            fh.write("\n".join([str(i) for i in test_set]))
        with open(train_file, "w") as fh:
            fh.write("\n".join([str(i) for i in train_set]))


### FUNCTIONS FOR CROSS-VALIDATION SETS


def make_cv_sets_for_model_evaluation(df: pd.DataFrame, path_to_holdouts: str, overwrite=True):
    """Make and save cross-validation sets for all taxonomic levels and all conditions.

    Used in experiments for model selection and evaluation.

    Args:
        df: DataFrame of data
        path_to_holdouts: Path to directory to save holdout sets
        overwrite: If True, saves file, overwriting any existing holdout sets. If False, does not write to files.
    Returns:
        None
    """
    for condition in CONDITIONS:
        cv_sets_dict = {}
        train_set, _ = load_train_and_test_sets(condition, path_to_holdouts)
        df_train = df.loc[train_set]
        for partition_rank in ["phylum", "class", "order", "family", "genus", "species"]:
            logging.info("Generating CV sets for %s at level %s", condition, partition_rank)
            cv_sets = make_cv_sets_by_phylogeny(genomes=df_train.index.tolist(), partition_rank=partition_rank, kfold=5)
            print(len(cv_sets))
            cv_sets_dict[partition_rank] = format_cv_sets_for_json(cv_sets)

        if overwrite is True:
            json.dump(cv_sets_dict, open(f"{path_to_holdouts}/{condition}_cv_sets.json", "w"))


def make_cv_sets_by_phylogeny(
    genomes: np.ndarray, partition_rank="family", kfold=5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """A method of making k-fold cross-validation sets composed
    of random taxa at particular taxonomic rank.

    This method is important at avoiding data leakage caused by the correlateion between
    traits and taxonomy. Within each fold, the method attempts to create partitions
    each with 1/kold proportion of genomes, but because taxa have uneven sizes,
    the actual size of the partition will vary between folds. This effect will be stronger
    at higher taxonomic levels where taxa are most uneven.

    Args:
        genomes: List of genome accessions
        partition_rank: Taxonomic rank to partition random genomes
        kfold: Number of folds to make
    Returns:
        List of tuples of training and validation set arrays

    """

    partitioner = PartitionTaxa(
        taxonomy=TaxonomyGTDB(),
        partition_rank=partition_rank,
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

    logging.info("Fold sizes for k=%i: %s", kfold, ", ".join([f"{f:.2f}" for f in fold_sizes]))
    return cv_sets


def make_cv_sets_randomly(genomes: np.ndarray, kfold=5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """A method of making classic random k-fold cross-validation sets,
    not recommended for microbiological datasets.

    Args:
        genomes: List of genome accessions
        kfold: Number of folds to make
    Returns:
        List of tuples of training and validation set arrays
    """
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


def format_cv_sets_for_json(cv_sets: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[List[int], List[int]]]:
    """Convert CV sets to JSON serializable format

    Args:
        cv_sets: List of tuples of training and validation set arrays
    Returns:
        List of tuples of training and validation set lists, with indices converted to ints
    """
    cv_sets_serializable = []
    for training_indices, validation_indices in cv_sets:
        serializable_set = ([int(idx) for idx in training_indices], [int(idx) for idx in validation_indices])
        cv_sets_serializable.append(serializable_set)
    return cv_sets_serializable


def yield_cv_sets(cv_sets: List[Tuple[np.ndarray, np.ndarray]]):
    """Generator required for scikit-learn cross-validation functions"""
    for training_indices, validation_indices in cv_sets:
        yield training_indices, validation_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make holdout sets for model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training_data_filename",
        type=str,
        required=True,
        help="Path to training data TSV file",
    )

    parser.add_argument(
        "--path_to_holdouts",
        type=str,
        required=True,
        help="Path to directory to save holdout sets",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If True, saves file, overwriting any existing holdout sets. If False, does not write to files.",
    )

    args = parser.parse_args()

    if args.overwrite is None:
        args.overwrite = False
    return args


if __name__ == "__main__":
    args = parse_args()
    make_holdout_sets(
        df=load_training_data(args.training_data_filename),
        path_to_holdouts=args.path_to_holdouts,
        overwrite=args.overwrite,
        partition_size=0.20,
    )
    make_cv_sets_for_model_evaluation(
        df=load_training_data(args.training_data_filename),
        path_to_holdouts=args.path_to_holdouts,
        overwrite=args.overwrite,
    )
