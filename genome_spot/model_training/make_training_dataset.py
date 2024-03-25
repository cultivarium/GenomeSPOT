import argparse
import logging
import multiprocessing
from glob import glob
from pathlib import Path
from typing import (
    List,
    Tuple,
    Union,
)

import pandas as pd

from ..bioinformatics.genome import measure_genome_features
from ..genome_spot import save_results
from ..helpers import load_file_pairs_from_directory
from ..taxonomy.taxonomy import TaxonomyGTDB


logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

IGNORE_GENOMES = [
    "GCA_000875775",  # BacDive salinity decimal point error up
    "GCA_003751385",  # BacDive salinity decimal point error up
    "GCA_023078355",  # BacDive salinity decimal point error up
    "GCA_000956175",  # BacDive salinity decimal point error up
    "GCA_001971705",  # BacDive salinity decimal point error down
    "GCA_013415885",  # BacDive salinity decimal point error down
    "GCA_013415905",  # BacDive salinity decimal point error down
    "GCA_017352095",  # BacDive salinity decimal point error down
    "GCA_020105915",  # BacDive salinity decimal point error down
    "GCA_000174915",  # bug? 6 total proteins
    "GCA_905142475",  # bug? proteins double counted
    "GCA_905220785",  # bug? proteins double counted
]


### FUNCTIONS TO MEASURE GENOME FEATURES IN PARALLEL


def generate_inputs(input_list: List[Tuple[str, str, str]], output_dir: str):
    """Generator to yield inputs in multiprocessing pool"""
    for n, (genome_accession, faa_path, fna_path) in enumerate(input_list):
        output_prefix = f"{output_dir}/{genome_accession}"
        yield n, Path(faa_path), Path(fna_path), output_prefix


def process_measure_genome_features(inputs: Tuple[int, str, str, str]):
    """Function to be called in multiprocessing pool. Measures genome features and saves to file."""
    n, faa_path, fna_path, output_prefix = inputs
    try:
        genome_features = measure_genome_features(faa_path=faa_path, fna_path=fna_path)
    except:
        genome_features = {}
    save_results(
        predictions=None, genome_features=genome_features, output_prefix=output_prefix, save_genome_features=True
    )
    if n % 1000 == 0:
        logger.info("%i features calculated", n)
    return inputs


def pool_measure_genome_features(
    directory: str, suffix_fna: str, suffix_faa: str, output_dir: str, processes: Union[int, None]
) -> list:
    """Use multiprocessing to measure genome features in parallel.

    Args:
        directory (str): directory containing genome files (pairs of DNA and protein FASTAs)
        suffix_fna (str): suffix of DNA FASTA files
        suffix_faa (str): suffix of protein FASTA files
        output_dir (str): directory to save genome features
        processes (int): number of parallel processes
    Returns:
        outputs (list): list of inputs provided to process_measure_genome_features
    """

    # Set inputs

    logger.info("Looking for %s and %s files in%s", suffix_fna, suffix_faa, directory)
    input_list, n_missing_files = load_file_pairs_from_directory(
        directory, suffix_fna=suffix_fna, suffix_faa=suffix_faa
    )
    filepath_gen = generate_inputs(input_list, output_dir)
    logger.info("Found both DNA and protein FASTA for %i genomes", len(input_list))
    logger.info("MISSING %i files (DNA or protein)", n_missing_files)

    workers = int(processes)
    if workers is None:
        workers = multiprocessing.cpu_count() - 1
    logging.info("Measuring %i genomes with %i CPUs", len(input_list), workers)

    # Measure
    with multiprocessing.Pool(workers) as p:
        pipeline_gen = p.map(process_measure_genome_features, filepath_gen)
        outputs = list(pipeline_gen)
    logger.info("Measured %i genomes", len(outputs))

    return outputs


### FUNCTIONS TO CREATE DATAFRAME


def load_features_to_dataframe(features_dir: str) -> pd.DataFrame:
    """Load features from genomes to a dataframe

    Args:
        features_dir (str): directory containing genome features
    Returns:
        df_features (pd.DataFrame): dataframe with genome features
    """
    sers = []
    for filename in glob(features_dir + "/*.features.json"):
        try:
            ser_genome = load_features_json_to_df(filename)
            sers.append(ser_genome)
        except:
            pass
    df_features = pd.concat(sers, axis=1).T

    # Missing data should be 0
    df_features = df_features.fillna(0.0)
    # Join localization and variable together
    renamed_cols = ["_".join(ser) for ser in df_features]
    df_features.columns = df_features.columns.droplevel()
    df_features.columns = renamed_cols

    return df_features


def load_features_json_to_df(filename: str) -> pd.DataFrame:
    """Load features from a genome to a dataframe"""
    sers = []
    json_df = pd.read_json(filename)
    for col in json_df.columns:
        ser = json_df.loc[:, col]
        accession = filename.split("/")[-1].split(".")[0]
        ser.name = accession
        index_tuples = list(zip([col] * len(ser), ser.index.tolist()))
        index = pd.MultiIndex.from_tuples(index_tuples, names=["localization", "variable"])
        ser.index = index
        sers.append(ser)

    df_genome = pd.concat(sers, axis=0)
    return df_genome


def load_target_dataframe(trait_data_tsv: str) -> pd.DataFrame:
    """Load trait data from a TSV file to a dataframe

    Args:
        trait_data_tsv (str): path to data computed by ComputeBacDiveTraits

    Returns:
        df_targets (pd.DataFrame): dataframe with trait data
    """
    df_targets = pd.read_csv(trait_data_tsv, sep="\t", index_col=0)
    return df_targets


def qc_features_dataframe(
    input_df_features: pd.DataFrame,
    min_coding_density: float = 0.60,
    max_coding_density: float = 1.0,
    use_preset_to_ignore_genomes: bool = True,
) -> pd.DataFrame:
    """Quality control of features dataframe by removing genomes with aberrant coding density
    and, optionally, genomes in a preset list defined in the script.
    """
    df_features = input_df_features.copy()
    below_coding_density_filter = df_features["all_protein_coding_density"] < min_coding_density
    above_coding_density_filter = df_features["all_protein_coding_density"] > max_coding_density
    genomes_to_drop = df_features[below_coding_density_filter | above_coding_density_filter].index.tolist()
    if use_preset_to_ignore_genomes:
        genomes_to_drop += IGNORE_GENOMES
    genomes_to_drop = list(set(genomes_to_drop).intersection(df_features.index))
    logging.info("Ignoring genomes: %s", ", ".join(genomes_to_drop))
    return df_features.drop(genomes_to_drop)


def qc_targets_dataframe(input_df_targets: pd.DataFrame) -> pd.DataFrame:
    """Quality control of targets dataframe by formatting values and columns"""

    # Cleanup values and columns
    df_targets = input_df_targets.copy()
    df_targets["ncbi_accession"] = [acc.split(".")[0] for acc in df_targets.index]
    df_targets = df_targets[~df_targets["ncbi_accession"].isnull()].set_index("ncbi_accession")
    quantitative_vars = [
        col for col in df_targets.columns if any([attr in col for attr in ["_optimum", "_min", "_max"]])
    ]
    df_targets.loc[:, quantitative_vars] = df_targets.loc[:, quantitative_vars].astype(float)
    df_targets = df_targets.rename(columns={"species": "ncbi_species"})

    # Add taxonomy
    taxonomy = TaxonomyGTDB()
    for taxlevel, index in taxonomy.indices.items():
        df_targets[taxlevel] = df_targets.index.map({k: v[index] for k, v in taxonomy.taxonomy_dict.items()})

    return df_targets


def make_training_df(features_dir: str, trait_data_tsv: str) -> pd.DataFrame:
    """Load features JSONs and target data to create a training dataframe

    Args:
        features_dir (str): directory containing genome features
        trait_data_tsv (str): path to trait data produced using
            the script download_training_data.py
    Returns:
        df (pd.DataFrame): dataframe with features and targets ready for training
    """
    df_features = load_features_to_dataframe(features_dir)
    df_features = qc_features_dataframe(df_features)
    df_targets = load_target_dataframe(trait_data_tsv=trait_data_tsv)
    df_targets = qc_targets_dataframe(df_targets)
    df = df_features.join(df_targets, how="inner")
    return df


def make_training_dataset(
    downloaded_traits: str,
    genomes_dir: str,
    suffix_fna: str,
    suffix_faa: str,
    output_features_dir: str,
    output_tsv: str,
    processes: Union[int, None] = None,
    skip_measure_features: bool = False,
) -> pd.DataFrame:
    """Make a training dataset by measuring genome features and joining
    that data with downloaded trait data

    Args:
        downloaded_traits (str): path to the trait data TSV downloaded from BacDive using download_training_data.py
        genomes_dir (str): path to directory containing subdirectories each with one genome FASTA and one protein FASTA
        suffix_fna (str): suffix of genome FASTA files, default .fna.gz
        suffix_faa (str): suffix of protein FASTA files, default .faa.gz
        output_features_dir (str): directory to save with genome feature files <genome_accession>.features.json
        output_tsv (str): path to write the training data TSV file
        processes (int): number of parallel processes (default=4)
        skip_measure_features (bool): whether to skip measuring genome features (recommended if already computed)
    Returns:
        df (pd.DataFrame): dataframe with features and targets
    """
    # Measure genome features in parallel
    if skip_measure_features is False:
        pool_measure_genome_features(genomes_dir, suffix_fna, suffix_faa, output_features_dir, processes)

    # Join features with downloaded trait data
    df = make_training_df(output_features_dir, downloaded_traits)
    df.to_csv(output_tsv, sep="\t")
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make holdout sets for model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-sfaa", required=True, help="Suffix of protein FASTA files, default .faa.gz")
    parser.add_argument("-sfna", required=True, help="Suffix of genome FASTA files, default .fna.gz")
    parser.add_argument(
        "-d",
        "--genomes-directory",
        help="Path to directory containing subdirectories each with one genome FASTA and one protein FASTA",
    )
    parser.add_argument(
        "-f",
        "--features-directory",
        help="Directory to save with genome feature files <genome_accession>.features.json",
        required=True,
    )
    parser.add_argument(
        "--downloaded-traits",
        type=str,
        required=True,
        help="Path to the trait data TSV downloaded from BacDive using download_training_data.py",
    )

    parser.add_argument(
        "-t",
        "--tsv-output",
        type=str,
        required=True,
        help="Path to write the training data TSV file",
    )

    parser.add_argument("-p", "--processes", help="Number of parallel processes (default=4)", default=4, required=False)

    parser.add_argument(
        "--skip-measure-features",
        action="store_true",
        help="Whether to skip measuring genome features (recommended if already computed)",
        default=False,
        required=False,
    )

    args = parser.parse_args()
    if args.features_directory.endswith("/"):
        args.features_directory = args.features_directory[:-1]

    return args


if __name__ == "__main__":
    args = parse_args()
    make_training_dataset(
        downloaded_traits=args.downloaded_traits,
        genomes_dir=args.genomes_directory,
        suffix_fna=args.sfna,
        suffix_faa=args.sfaa,
        output_features_dir=args.features_directory,
        output_tsv=args.tsv_output,
        processes=args.processes,
        skip_measure_features=args.skip_measure_features,
    )
