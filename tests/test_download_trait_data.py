# pylint: disable=missing-docstring
import json
from pathlib import Path

import pandas as pd
from genome_spot.model_training.download_trait_data import (
    ComputeBacDiveTraits,
    load_targets_to_dataframe,
)


cwd = Path(__file__).resolve().parent

BACDIVE_DATA = f"{cwd}/test_data/test_bacdive_data.json"
TRAIT_EXAMPLE = f"{cwd}/test_data/GCA_003143535.1.traits.json"
TRAIT_DATA = f"{cwd}/test_data/test_trait_data.tsv"


class TestComputeBacDiveTraits:

    def test_compute_trait_data(self):
        with open(BACDIVE_DATA, "r") as f:
            bacdive_data = json.loads(f.read())
        with open(TRAIT_EXAMPLE, "r") as f:  # traits for a single strain
            expected_values = json.loads(f.read())
        bacdive_entry = bacdive_data["166247"]  # downloaded data for a single strain
        strain_traits = ComputeBacDiveTraits(bacdive_entry).compute_trait_data()
        assert strain_traits == expected_values

    def test_salinity_retained(self):
        # It's easy to accidentally remove salinity = 0 values because
        # the optimum is the minimum
        with open(BACDIVE_DATA, "r") as f:
            bacdive_data = json.loads(f.read())
        bacdive_entry = bacdive_data["164638"]
        strain_traits = ComputeBacDiveTraits(bacdive_entry).compute_trait_data()
        assert strain_traits["salinity_optimum"] == 0.0
        assert strain_traits["use_salinity"] == True


def test_load_targets_to_dataframe():
    traits_df = load_targets_to_dataframe(BACDIVE_DATA)
    expected_df = pd.read_csv(TRAIT_DATA, sep="\t", index_col=0).set_index("ncbi_accession", drop=False)
    assert set(traits_df.index).difference(expected_df.index) == set()
    assert set(traits_df.columns).difference(expected_df.columns) == set()
