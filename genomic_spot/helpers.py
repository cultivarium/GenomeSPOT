"""Generally useful functions"""

import json
from collections import Counter
from itertools import groupby
from re import L
from typing import (
    IO,
    Dict,
    List,
    Tuple,
)

import numpy as np
import pandas as pd


def count_kmers(sequence: str, k: int) -> Dict[str, float]:
    """Returns counts of every observed k-mer at specific k.

    Args:
        sequence: Sequence, protein or nucleotide
        k: Length of string

    Returns:
        Dictionary of k-mer counts e.g. {'AA' : 2, ...}
    """
    kmers_count = Counter([sequence[i : i + k] for i in range(len(sequence) - k + 1)])
    return dict(kmers_count)


def iterate_fasta(fasta_file: IO):
    """Iterable yielding FASTA header and sequence

    Modified from: https://www.biostars.org/p/710/

    Args:
        fasta_file: File object for FASTA file

    Yields:
        headerStr: Header following without >
        seq: FASTA sequence
    """
    faiter = (x[1] for x in groupby(fasta_file, lambda line: line[0] == ">"))
    for header in faiter:
        headerStr = header.__next__()[1:].strip()
        seq = "".join(s.strip() for s in faiter.__next__())
        yield (headerStr, seq)


def rename_condition_to_variable(condition, attribute="optimum"):
    """Commonly used to turn a condition e.g. 'temperature'
    to a variable e.g. 'temperature_optimum'"""
    if condition == "oxygen":
        return "oxygen"
    else:
        return condition + "_" + attribute


def prepend_features(features, prefices):
    """useful for assigning localization to features"""
    return [f"{prefix}_{feature}" for prefix in prefices for feature in features]


def load_cv_sets(condition, path_to_holdouts, taxlevel="family") -> List[Tuple[np.ndarray, np.ndarray]]:
    """Loads cross-validation sets to list of (training set, validation set)

    Args:
        condition (str): condition e.g. 'temperature'
        path_to_holdouts (str): path to holdouts directory
        taxlevel (str): taxonomic level of holdouts e.g. 'family'
    Returns:
        cv_sets: list of (training set, validation set) for each fold
    """
    cv_sets_dict_file = f"{path_to_holdouts}/{condition}_cv_sets.json"
    with open(cv_sets_dict_file, "r") as fh:
        cv_sets_dict = json.loads(fh.read())

    cv_sets = []
    for cv_set in cv_sets_dict[taxlevel]:
        cv_sets.append((np.array(cv_set[0]), np.array(cv_set[1])))
    return cv_sets


def load_train_and_test_sets(condition: str, path_to_holdouts: str) -> Tuple[List[str], List[str]]:
    """Loads training and test sets for a condition

    Args:
        condition (str): condition e.g. 'temperature'
        path_to_holdouts (str): path to holdouts directory
    Returns:
        training_set (List[str]): list of training set accession numbers
        test_set (List[str]): list of test set accession numbers
    """
    training_set_filename = f"{path_to_holdouts}/train_set_{condition}.txt"
    with open(training_set_filename, "r") as fh:
        training_set = [line.strip() for line in fh.readlines()]
    test_set_filename = f"{path_to_holdouts}/test_set_{condition}.txt"
    with open(test_set_filename, "r") as fh:
        test_set = [line.strip() for line in fh.readlines()]
    return training_set, test_set


def split_train_and_test_data(
    df: pd.DataFrame, condition: str, path_to_holdouts: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataframe into training and test sets

    Args:
        df (pd.DataFrame): dataframe
        condition (str): condition e.g. 'temperature'
        path_to_holdouts (str): path to holdouts directory
    Returns:
        df_train (pd.DataFrame): training set data
        df_test (pd.DataFrame): test set data
    """
    training_set, test_set = load_train_and_test_sets(condition, path_to_holdouts)
    df_train = df.loc[list(set(training_set).intersection(set(df.index))), :]
    df_test = df.loc[list(set(test_set).intersection(set(df.index))), :]
    return df_train, df_test


def load_training_data(training_data_filename) -> pd.DataFrame:
    """Loads training data

    Args:
        training_data_filename (str): path to training data
    Returns:
        training_df (pd.DataFrame): training data
    """
    training_df = pd.read_csv(training_data_filename, sep="\t", index_col=0)
    return training_df
