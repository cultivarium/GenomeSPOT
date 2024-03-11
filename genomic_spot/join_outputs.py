"""Simple helper to join outputs of individual genome predictions into a single TSV and/or JSON."""

import argparse
import json
from glob import glob
from typing import (
    List,
    Tuple,
)

import pandas as pd


def join_outputs(outdir: str, write_to_tsv: bool = True, write_to_dict: bool = False) -> None:
    """Join outputs of individual genome predictions into a single TSV and/or JSON."""
    output_files = get_output_filepaths(outdir)

    if write_to_tsv is True:
        single_tsv = convert_outputs_to_single_tsv(output_files)
        single_tsv.to_csv("all.predictions.tsv", sep="\t")

    if write_to_dict is True:
        output_dict = convert_outputs_to_nested_dict(output_files)
        with open("all.predictions.json", "w") as f:
            json.dump(output_dict, f, indent=4)


def get_output_filepaths(outdir: str) -> List[str]:
    """Get filepaths for all prediction.tsv files in a directory."""
    output_files = glob(f"{outdir}/*.predictions.tsv")
    return output_files


def load_output_tsv(tsv: str) -> Tuple[str, pd.DataFrame]:
    """Load a single prediction.tsv file into a pandas DataFrame."""
    accession = tsv.split("/")[-1].split(".")[0]
    predictions_df = pd.read_csv(tsv, sep="\t", index_col=0)
    return accession, predictions_df


def convert_outputs_to_single_tsv(output_files: List[str]) -> pd.DataFrame:
    """Convert multiple prediction.tsv files into a single DataFrame."""
    dfs = []
    for tsv in output_files:
        try:
            accession, predictions_df = load_output_tsv(tsv)
            melted_df = pd.melt(
                predictions_df.reset_index().rename(columns={"value": "prediction"}),
                id_vars=["target"],
                value_vars=["prediction", "error", "units", "is_novel", "warning"],
            )
            melted_df = melted_df.set_index(["target", "variable"]).rename(columns={"value": accession}).T
            dfs.append(melted_df)
        except:
            pass
    single_tsv = pd.concat(dfs, axis=0)
    return single_tsv


def convert_outputs_to_nested_dict(output_files: List[str]) -> dict:
    """Convert multiple prediction.tsv files into a nested dictionary."""
    output_dict = {}
    for tsv in output_files:
        try:
            accession, predictions_df = load_output_tsv(tsv)
            output_dict[accession] = {}
            predictions_dict = predictions_df.T.to_dict()
            for condition in ["temperature", "ph", "salinity"]:
                condition_dict = {}
                for attr in ["optimum", "min", "max"]:
                    target = f"{condition}_{attr}"
                    condition_dict[attr] = predictions_dict[target]
                    condition_dict[attr].pop("is_novel", None)
                    condition_dict[attr].pop("units", None)
                    if attr == "optimum":
                        condition_dict["is_novel"] = predictions_dict[target].get("is_novel", None)
                        condition_dict["units"] = predictions_dict[target].get("units", None)

                output_dict[accession][condition] = condition_dict
        except:
            pass
    return output_dict


def parse_args():

    parser = argparse.ArgumentParser(description="Join outputs of individual genome predictions")
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to directory containing *.prediction.tsv files",
    )
    parser.add_argument(
        "--write-to-tsv",
        action="store_true",
        default=False,
        help="If flag used, save output TSV to all.predictions.tsv",
    )
    parser.add_argument(
        "--write-to-json",
        action="store_true",
        default=False,
        help="If flag used, save output JSON to all.predictions.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    join_outputs(
        outdir=args.dir,
        write_to_tsv=args.write_to_tsv,
        write_to_dict=args.write_to_json,
    )
