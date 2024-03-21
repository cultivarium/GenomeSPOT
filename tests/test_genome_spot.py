# pylint: disable=missing-docstring
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import sklearn
from genome_spot.bioinformatics.genome import load_genome_features
from genome_spot.genome_spot import GenomeSPOT


cwd = Path(__file__).resolve().parent


CONTIG_FASTA = f"{cwd}/test_data/GCA_000172155.1_ASM17215v1_genomic.fna.gz"
PROTEIN_FASTA = f"{cwd}/test_data/GCA_000172155.1_ASM17215v1_protein.faa.gz"
GENOME_FEATURES_JSON = f"{cwd}/test_data/GCA_000172155.features.json"
PREDICTIONS_JSON = f"{cwd}/test_data/GCA_000172155.predictions.json"
INSTRUCTIONS_JSON = f"{cwd}/test_data/instructions.json"
MODEL_FILE = f"{cwd}/test_data/oxygen.joblib"
CONDITIONS = ["oxygen", "temperature", "salinity", "ph"]
LOCALIZATIONS = ["all", "extracellular_soluble", "intracellular_soluble", "membrane", "diff_extra_intra"]


class TestGenomeSPOT:

    def test_sklearn_version(self):
        # Model must have been trained with the same version of scikit-learn
        model = joblib.load(MODEL_FILE)
        sklearn_version = model.__getstate__()["_sklearn_version"]
        assert sklearn_version == sklearn.__version__

    def test_units(self):
        genome_spot = GenomeSPOT()
        assert all([k in CONDITIONS for k in genome_spot.UNITS.keys()])

    def test_prediction_bounds(self):
        genome_spot = GenomeSPOT()
        for v in genome_spot.PREDICTION_BOUNDS.values():
            assert v[0] < v[1]
        assert all([k in CONDITIONS for k in genome_spot.PREDICTION_BOUNDS.keys()])

    def test_load_and_format_genome_features(self):

        expected_arr = np.array(
            [
                [
                    0.10206504,
                    0.01126556,
                    0.05028364,
                    0.06152117,
                    0.03772187,
                    0.07959808,
                    0.02449539,
                    0.04365775,
                    0.04550563,
                    0.10175385,
                    0.02434063,
                    0.02880634,
                    0.05701037,
                    0.03695734,
                    0.06506852,
                    0.06170794,
                    0.05536257,
                    0.07260104,
                    0.01658095,
                    0.02404187,
                ]
            ]
        )
        genome_features = load_genome_features(GENOME_FEATURES_JSON)
        genome_spot = GenomeSPOT()
        features = genome_spot.load_instructions(INSTRUCTIONS_JSON)["oxygen"]["features"]
        input_arr = genome_spot.genome_features_to_input_arr(features=features, genome_features=genome_features)
        assert all([k in genome_features.keys() for k in LOCALIZATIONS])
        assert input_arr == pytest.approx(expected_arr)

    def test_predict_target_value(self):
        genome_spot = GenomeSPOT()
        genome_features = load_genome_features(GENOME_FEATURES_JSON)
        model = joblib.load(MODEL_FILE)
        features = genome_spot.load_instructions(INSTRUCTIONS_JSON)["oxygen"]["features"]
        X = genome_spot.genome_features_to_input_arr(features, genome_features)
        predictions = genome_spot.predict_target_value(
            target="oxygen",
            X=X,
            model=model,
            method="predict_proba",
            error_model=None,
            novelty_model=None,
        )

        assert all([k in ["value", "error", "is_novel", "warning", "units"] for k in predictions.keys()])
        # check there are values as expected
        assert predictions["value"] == "tolerant"
        assert predictions["error"] == pytest.approx(0.95, abs=0.3)
        assert predictions["units"] == "probability"

    def test_format_to_tsv(self):
        predictions = json.loads(open(PREDICTIONS_JSON, "r").read())
        genome_spot = GenomeSPOT()
        tsv = genome_spot.format_to_tsv(predictions)
        tsv_header = tsv.split("\n")[0].split("\t")
        assert tsv_header == ["target", "value", "error", "units", "is_novel", "warning"]
        for line in tsv.split("\n"):
            assert len(line.split("\t")) == 6
