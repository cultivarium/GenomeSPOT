"""Core script to predict conditions from a genome"""

import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from typing import (
    Dict,
    Tuple,
    Union,
)

import joblib
import numpy as np
import pandas as pd
from genome import Genome


# Used to bound predictions to sensical values/ranges from training
PREDICTION_BOUNDS = {
    "temperature": (0, 105),
    "ph": (0.5, 14),
    "salinity": (0, 37),
    "oxygen": (0, 1),
}


def predict_from_genome(genome_features: dict, path_to_models: str) -> Dict[str, dict]:
    """Predicts growth conditions from genome features for all models specified
    in the provided instructions.

    For continuous growth conditions, a min, max, and optimum are computed. For
    binary classifications, the probability of the second class is returned.

    Args:
        genome_features: nested dict of all genome features computed by Genome
        path_to_models: path to directory with models
    Returns:
        predictions: nested dict of target and predicted value and upper and lower confidence
            intervals
    """
    predictions = defaultdict(dict)
    instructions = load_instructions(f"{path_to_models}/instructions.json")
    for condition in instructions.keys():
        if condition in ["ph", "temperature", "salinity"]:
            features = instructions[condition]["features"]
            X = genome_features_to_input_arr(features, genome_features)
            for attribute in [
                "optimum",
                "max",
                "min",
            ]:
                target = f"{condition}_{attribute}"
                model = joblib.load(f"{path_to_models}/{target}.joblib")
                predictions[target] = predict_target_value(target=target, X=X, model=model)
        elif condition == "oxygen":
            target = condition
            features = instructions[condition]["features"]
            model = joblib.load(f"{path_to_models}/{target}.joblib")
            X = genome_features_to_input_arr(features, genome_features)
            predictions[target] = predict_target_value(target=target, X=X, model=model, method="predict_proba")
    return predictions


def load_instructions(instructions_filename: str) -> dict:
    """Load pipeline and features stored in instruction file

    The instructions contain instructions for condition (e.g. pH)
    for each target keyed by [condition]["features"]

    Args:
        instructions_filename: filepath to insructions file
    """
    with open(instructions_filename) as fh:
        instructions = json.loads(fh.read())
    return instructions


def genome_features_to_input_arr(features: list, genome_features: dict) -> np.ndarray:
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
        for localization, feat_dict in genome_features.items()
        for feat, value in feat_dict.items()
    }
    try:
        assert len(features) == len(set(features).intersection(set(flat_genome_features.keys())))
    except Exception as exc:
        missing_features = set(features).difference(set(flat_genome_features.keys()))
        raise ValueError(f"Features were provided that are not found in genome features: {missing_features}") from exc
    X = np.array([flat_genome_features.get(x) for x in features]).reshape(1, -1)
    return X


def predict_target_value(X: np.ndarray, model, target: str, method: str = "predict") -> Dict[str, float]:
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
    if method == "predict":
        y_pred = model.predict(X)[0]
    elif method == "predict_proba":
        y_pred = model.predict_proba(X[0, :].reshape(1, -1))[:, 1][0]

    y_pred, warning = check_prediction_range(y_pred, target)

    # TO-DO: compute confidence intervals
    lower_ci = None
    upper_ci = None
    prediction_dict = {
        "value": y_pred,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "warning": warning,
    }
    return prediction_dict


def check_prediction_range(y_pred: float, target: str) -> Tuple[float, Union[str, None]]:
    """If the prediction is above or below the bounds set in this
    script, a warning flag is added to the output and the value is
    set to either the max or min, whichever was exceeded.
    """
    min_pred, max_pred = PREDICTION_BOUNDS[target.split("_")[0]]
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


def load_genome_features(features_json: str) -> Dict[str, dict]:
    """Loads a JSON containing features, if available"""
    logging.info("Loading existing genome features from %s", features_json)
    with open(features_json, "r", encoding="utf-8") as fh:
        genome_features = json.loads(fh.read())
    return genome_features


def measure_genome_features(faa_path: str, fna_path: str) -> Dict[str, dict]:
    """Measure features from the provided genome files"""
    logging.info(
        "Measuring features from:\n\t%s\n\t%s",
        fna_path,
        faa_path,
    )
    genome_features = Genome(
        contig_filepath=fna_path,
        protein_filepath=faa_path,
    ).measure_genome_features()
    return genome_features


def predict_physicochemistry(
    fna_path: str,
    faa_path: str,
    path_to_models: str,
    features_json: str = None,
    skip_prediction: bool = False,
) -> Tuple[dict, dict]:
    """Main function for predicting traits from genome sequences (DNA and protein)."""
    if features_json is not None:
        genome_features = load_genome_features(features_json)
    else:
        genome_features = measure_genome_features(faa_path, fna_path)

    # Predict
    if skip_prediction is False:
        logging.info("Predicting growth conditions")
        predictions = predict_from_genome(genome_features, path_to_models)
    else:
        logging.info("Skipping prediction of growth conditions as `run_prediction` was set to False ")
        predictions = {}

    return predictions, genome_features


def save_results(predictions: dict, genome_features: dict, output_prefix: str, save_genome_features: bool):
    if output_prefix is not None and save_genome_features is True:
        intermediate_output = str(output_prefix) + ".features.json"
        logging.info("Saving intermediate to %s", intermediate_output)
        json.dump(genome_features, open(intermediate_output, "w", encoding="utf-8"))
    if output_prefix is not None and predictions is not None:
        output = str(output_prefix) + ".predictions.tsv"
        logging.info("Saving output to: %s", output)
        save_predictions(predictions, output_tsv=output)


def save_predictions(predictions: dict, output_tsv: str) -> str:
    """Saves ouput of `predict_with_all_models` to a tab-separated table"""
    with open(output_tsv, "w", encoding="utf-8") as fh:
        lines = []
        lines.append("\t".join(["target", "value", "lower_ci", "upper_ci"]))
        for target in sorted(predictions):
            values = predictions[target]
            line = [
                target,
                str(values["value"]),
                str(values["lower_ci"]),
                str(values["upper_ci"]),
            ]
            lines.append("\t".join(line))
        fh.write("\n".join(lines))
        return output_tsv


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


def validate_args(args: ArgumentParser):
    if (args.contigs is None or args.proteins is None) and args.genome_features is None:
        raise ValueError(
            """User must provide either files for contigs and 
            proteins or a precomputed genome features file"""
        )


def main(args: ArgumentParser):
    predictions, genome_features = predict_physicochemistry(
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
