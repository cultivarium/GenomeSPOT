"""Core script to predict conditions from a genome"""

import json
import logging
from argparse import (
    ArgumentParser,
    Namespace,
)
from collections import defaultdict
from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
)

import joblib
import numpy as np
import pandas as pd

from .bioinformatics.genome import (
    load_genome_features,
    measure_genome_features,
)


logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", format="%(asctime)s %(levelname)s %(message)s")


class GenomeSPOT:
    """Main class for predicting growth conditions from genome features

    Currently supported growth condition predictions are:
    - Oxygen tolerance (classification)
    - Temperature (regression)
    - pH (regression)
    - Salinity (regression)

    An estimate of error is provided for each prediction. For oxygen, this
    is the probability of the prediction. For continuous variables, the error
    is more appoximate: it is the RMSE of predictions close each predicted value.

    A novelty detection model is used to determine if a genome
    has unusual features compared to the training set. Novel genomes are less
    likely to be predicted accurately.

    Example usage:

    ```python
    from genome_spot.genome_spot import GenomeSPOT
    from genome_spot.bioinformatics.genome import measure_genome_features
    genome_features = measure_genome_features(faa_path, fna_path)
    predictions = GenomeSPOT().predict_from_genome(genome_features, path_to_models)
    tsv = GenomeSPOT().format_to_tsv(predictions)
    print(tsv)
    ```
    """

    # Used to bound predictions to sensical values/ranges from training
    PREDICTION_BOUNDS = {
        "temperature": (0, 105),
        "ph": (0.5, 14),
        "salinity": (0, 37),
        "oxygen": (0, 1),
    }

    UNITS = {"temperature": "C", "ph": "pH", "salinity": "% w/v NaCl", "oxygen": "probability"}

    def __init__(self):
        pass

    def predict_from_genome(
        self,
        genome_features: Dict[str, dict],
        path_to_models: str,
    ) -> Dict[str, dict]:
        """
        Predicts growth conditions from genome features for all models specified
        in the provided instructions. For continuous growth conditions, a min, max,
        and optimum are computed. For classification of oxygen, a genome is classified
        as a tolerant (aerobe or facultative anaerobe) or intolerant (obligate anaerobe).

        Args:
            genome_features: nested dict of all genome features computed by Genome
            path_to_models: path to directory with models
        Returns:
            predictions: nested dict of each target's predicted value, error, novelty, warning, and units
        """
        predictions = defaultdict(dict)
        instructions = self.load_instructions(f"{path_to_models}/instructions.json")
        for condition in instructions.keys():
            novelty_model = joblib.load(f"{path_to_models}/novelty_{condition}.joblib")
            if condition in ["ph", "temperature", "salinity"]:
                features = instructions[condition]["features"]
                for attribute in [
                    "optimum",
                    "max",
                    "min",
                ]:
                    X = self.genome_features_to_input_arr(features, genome_features)
                    target = f"{condition}_{attribute}"
                    model = joblib.load(f"{path_to_models}/{target}.joblib")
                    error_model = joblib.load(f"{path_to_models}/error_{target}.joblib")
                    predictions[target] = self.predict_target_value(
                        target=target, X=X, model=model, error_model=error_model, novelty_model=novelty_model
                    )
            elif condition == "oxygen":
                target = condition
                features = instructions[condition]["features"]
                model = joblib.load(f"{path_to_models}/{target}.joblib")
                X = self.genome_features_to_input_arr(features, genome_features)
                predictions[target] = self.predict_target_value(
                    target=target,
                    X=X,
                    model=model,
                    method="predict_proba",
                    error_model=None,
                    novelty_model=novelty_model,
                )
        return predictions

    def load_instructions(self, instructions_filename: str) -> dict:
        """Load pipeline and features stored in instruction file

        The instructions contain instructions for condition (e.g. pH)
        for each target keyed by [condition]["features"]

        Args:
            instructions_filename: filepath to insructions file
        """
        with open(instructions_filename, "r") as fh:
            instructions = json.loads(fh.read())
        return instructions

    def genome_features_to_input_arr(self, features: list, genome_features: dict) -> np.ndarray:
        """Creates an array that can be used as input for predictions.

        Genome features are recorded in a nested dictionary of
        format `{localization_A : {feature_1 : value_1}}`. This function
        makes a non-nested dictionary where localization prepends each feature,
        matches features, and provides them in the shape needed by scikitlearn models.

        Args:
            features: list of features to be used by model
            genome_features: nested dict of all genome features computed by Genome
        Returns:
            X: An array of feature values for one genome
        """
        flat_genome_features = {
            f"{localization}_{feat}": value
            for localization, feat_dict in sorted(genome_features.items())
            for feat, value in sorted(feat_dict.items())
        }
        X = np.array([flat_genome_features.get(x, np.nan) for x in features]).reshape(1, -1)
        return X

    def predict_target_value(
        self, X: np.ndarray, model, target: str, method: str = "predict", error_model=None, novelty_model=None
    ) -> Dict[str, float]:
        """Predicts a value and confidence intervals.

        Args:
            X: An array of feature values for one genome
            target: Name of the target variable, with a model at the specified location
            method: Regressions should be set to `predict` and and classifiers should be
                set to `predict_proba`
        Returns:
            prediction_dict: Dict containing the predicted value and upper and lower limit
                of confidence intervals
        """
        condition = target.replace("_optimum", "").replace("_min", "").replace("_max", "")
        y_pred = None
        units = self.UNITS[condition]
        error = None
        warning = None
        novelty = None

        if any(np.isnan(X[0])):
            # cannot predict
            warning = "genome missing features"
        else:
            if method == "predict":
                y_pred = model.predict(X)[0]
                y_pred, warning = self.check_prediction_range(y_pred, target)
                if error_model is not None:
                    error = self.predict_error(y_pred, error_model)
            elif method == "predict_proba":
                y_pred_prob = model.predict_proba(X[0, :].reshape(1, -1))[:, 1][0]
                y_pred, error = self.reformat_oxygen_prediction(y_pred_prob)

            if novelty_model is not None:
                novelty = self.predict_novelty(X, novelty_model)

        prediction_dict = {
            "value": y_pred,
            "error": error,
            "is_novel": novelty,
            "warning": warning,
            "units": units,
        }

        return prediction_dict

    def predict_error(self, y: float, error_arr: np.ndarray):
        """References an array where col 1 is a predicted value
        and col 2 is the RMSE of values predicted to be y +/- interval"""
        closest_reference = np.argmin(np.abs(error_arr[:, 0] - y))
        ref_y, ref_err = error_arr[closest_reference]
        return ref_err

    def predict_novelty(self, X: np.ndarray, novelty_model):
        """Predicts novelty using novelty detection model.

        Novelty detection compares a model to the training set to determine if
        a a new observation is within the range of the training set.
        The novelty detection algorithm (OneClassSVM) returns 1
        if not novel and -1 if novel. X must have the same features
        as the data the model was trained on.
        """
        is_novel = False if novelty_model.predict(X) == 1 else True
        return is_novel

    def check_prediction_range(self, y_pred: float, target: str) -> Tuple[float, Union[str, None]]:
        """If the prediction is above or below the bounds set in this
        script, a warning flag is added to the output and the value is
        set to either the max or min, whichever was exceeded.
        """
        min_pred, max_pred = self.PREDICTION_BOUNDS[target.split("_")[0]]
        if y_pred < min_pred:
            revised_y_pred = min_pred
            warning = "min_exceeded"
        elif y_pred > max_pred:
            revised_y_pred = max_pred
            warning = "max_exceeded"
        else:
            revised_y_pred = y_pred
            warning = None
        return revised_y_pred, warning

    def reformat_oxygen_prediction(self, y_pred_prob: float) -> Tuple[str, float]:
        """Reformat oxygen prediction to be more human-readable.
        Classifies as tolerant or intolerant and provides
        the probability of the prediction as the 'error' value.

        Args:
            y_pred_prob: probability of being tolerant
        """
        if y_pred_prob > 0.5:
            y_pred = "tolerant"
            error = y_pred_prob
        else:
            y_pred = "not tolerant"
            error = 1 - y_pred_prob
        return y_pred, error

    def format_to_tsv(self, predictions):
        """Formats predictions to a tab-separated table for output"""
        lines = []
        cols = ["target", "value", "error", "units", "is_novel", "warning"]
        lines.append("\t".join(cols))
        for target in sorted(predictions):
            values = predictions[target]
            line = [target] + [str(values[col]) for col in cols[1:]]
            lines.append("\t".join(line))
        return "\n".join(lines)


def run_genome_spot(
    fna_path: str,
    faa_path: str,
    path_to_models: str,
    features_json: Optional[str] = None,
    skip_prediction: bool = False,
) -> Tuple[dict, dict]:
    """Main function for predicting traits from genome sequences (DNA and protein).

    Args:
        fna_path: Path to a genome's contigs in FASTA format (.gz allowed)
        faa_path: Path to a genome's proteins in FASTA format (.gz allowed)
        path_to_models: Path to directory containing models
        features_json: Optional: Path to a precomputed intermediate genome features JSON file
        skip_prediction: If flag used, skips the prediction step. Useful if only genome features are desired

    Peturns:
        predictions: nested dict of target and predicted value and upper and lower confidence intervals
        genome_features: nested dict of all genome features computed by Genome
    """
    # Load or measure genome features
    if features_json is not None:
        genome_features = load_genome_features(features_json)
    else:
        genome_features = measure_genome_features(faa_path, fna_path)

    # Predict
    if skip_prediction is False:
        logging.info("Predicting growth conditions")
        predictions = GenomeSPOT().predict_from_genome(genome_features, path_to_models)
    else:
        logging.info("Skipping prediction of growth conditions as `run_prediction` was set to False ")
        predictions = {}

    return predictions, genome_features


def save_results(
    predictions: Optional[dict] = None,
    genome_features: Optional[dict] = None,
    output_prefix: Optional[str] = None,
    save_genome_features: bool = True,
):
    """Saves results to file if output_prefix is provided""" ""
    if output_prefix is not None:
        if save_genome_features is True:
            intermediate_output = str(output_prefix) + ".features.json"
            logging.info("Saving genome features to %s", intermediate_output)
            json.dump(genome_features, open(intermediate_output, "w", encoding="utf-8"))
        if predictions is not None:
            output = str(output_prefix) + ".predictions.tsv"
            logging.info("Saving output to: %s", output)
            tsv = GenomeSPOT().format_to_tsv(predictions)
            with open(output, "w", encoding="utf-8") as fh:
                fh.write(tsv)


def parse_args():
    parser = ArgumentParser(prog="MeasureGenome", description="Measure and predict properties from a genome")

    parser.add_argument(
        "-c",
        "--contigs",
        default=None,
        help="Path to a genome's contigs in FASTA format (.gz allowed)",
    )
    parser.add_argument(
        "-p",
        "--proteins",
        default=None,
        help="Path to a genome's proteins in FASTA format (.gz allowed)",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        default=None,
        required=False,
        help="Prefix for output file <prefix>.predictions.tsv. If None, no output files will be saved.",
    )
    parser.add_argument(
        "-m",
        "--models",
        default=None,
        help="Path to directory containing models",
    )
    parser.add_argument(
        "-g",
        "--genome-features",
        default=None,
        help="Optional: Path to a precomputed intermediate genome features JSON file",
    )
    parser.add_argument(
        "--save-genome-features",
        action="store_true",
        default=False,
        help="If flag used, save genome features to <output_prefix>.features.json",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        default=False,
        help="If flag used, skips the prediction step. Useful if only genome features are desired",
    )

    args = parser.parse_args()
    validate_args(args)

    return args


def validate_args(args: Namespace):
    if (args.contigs is None or args.proteins is None) and args.genome_features is None:
        raise ValueError(
            """User must provide either files for contigs and 
            proteins or a precomputed genome features file"""
        )


def main(args: Namespace):
    predictions, genome_features = run_genome_spot(
        fna_path=args.contigs,
        faa_path=args.proteins,
        features_json=args.genome_features,
        skip_prediction=args.skip_prediction,
        path_to_models=args.models,
    )

    save_results(
        predictions=predictions,
        genome_features=genome_features,
        output_prefix=args.output_prefix,
        save_genome_features=args.save_genome_features,
    )
    print(pd.DataFrame(predictions).T)


if __name__ == "__main__":
    args = parse_args()
    main(args)
