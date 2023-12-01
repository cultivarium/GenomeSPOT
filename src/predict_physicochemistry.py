"""Core script to predict conditions from a genome"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

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

ROOT_DIR = str(Path(__file__).resolve().parent.parent)


def load_instructions(instructions_filename: str):
    """Load pipeline and features stored in instruction file"""
    with open(instructions_filename) as fh:
        instructions = json.loads(fh.read())
    return instructions


def genome_features_to_input_arr(features: list, genome_features: dict) -> np.array:
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


def predict_target_value(X, target: np.array, method="predict") -> dict:
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
    model = joblib.load(f"{ROOT_DIR}/models/{target}.joblib")
    if method == "predict":
        y_pred = model.predict(X)[0]
    elif method == "predict_proba":
        y_pred = model.predict_proba(X[0, :].reshape(1, -1))[:, 1][0]

    # Constrain within allowed values
    min_pred, max_pred = PREDICTION_BOUNDS[target.split("_")[0]]
    if y_pred < min_pred:
        y_pred = min_pred
        warning = "min_exceeded"
    elif y_pred > max_pred:
        y_pred = max_pred
        warning = "max_exceeded"
    else:
        warning = None

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


def predict_with_all_models(genome_features: dict, instructions_filename: str) -> dict:
    """Predicts growth conditions from genome features for all models specified
    in the provided instructions.

    For continuous growth conditions, a min, max, and optimum are computed. For
    binary classifications, the probability of the second class is returned.

    Args:
        genome_features: nested dict of all genome features computed by Genome
        instructions_filename: path to file containing condition (e.g. pH) for each target
            keyed by [condition]["features"]
    Returns:
        predictions: nested dict of target and predicted value and upper and lower confidence
            intervals
    """
    predictions = defaultdict(dict)
    instructions = load_instructions(instructions_filename)
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
                predictions[target] = predict_target_value(target=target, X=X)
        elif condition == "oxygen":
            target = condition
            features = instructions[condition]["features"]
            X = genome_features_to_input_arr(features, genome_features)
            predictions[target] = predict_target_value(target=target, X=X, method="predict_proba")
    return predictions


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


def predict_physicochemistry(
    fna_path: str,
    faa_path: str,
    instructions_filename: str,
    features_json: str = None,
    skip_prediction: bool = False,
):
    """Main function for predicting traits from genome sequences (DNA and protein)."""

    # Load or measure genomic features
    if features_json is not None:
        logging.info("Loading existing genome features from %s", features_json)
        with open(features_json, "r", encoding="utf-8") as fh:
            genome_features = json.loads(fh.read())
    else:
        logging.info(
            "Measuring features from genome and proteins:\n\t%s\n\t%s",
            fna_path,
            faa_path,
        )
        genome_features = Genome(
            contig_filepath=fna_path,
            protein_filepath=faa_path,
        ).genome_metrics()

    # Predict
    if skip_prediction is False:
        logging.info("Predicting growth conditions")
        predictions = predict_with_all_models(genome_features, instructions_filename)
    else:
        logging.info("Skipping prediction of growth conditions as `run_prediction` was set to False ")
        predictions = {}

    return predictions, genome_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MeasureGenome", description="Measure and predict properties from a genome")

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

    if (args.contigs is None or args.proteins is None) and args.genome_features is None:
        raise ValueError(
            """User must provide either files for contigs and 
            proteins or a precomputed genome features file"""
        )

    predictions, genome_features = predict_physicochemistry(
        fna_path=args.contigs,
        faa_path=args.proteins,
        features_json=args.genome_features,
        skip_prediction=args.skip_prediction,
        instructions_filename=f"{ROOT_DIR}/models/instructions.json",
    )

    if args.output_prefix is not None and args.save_genome_features is True:
        intermediate_output = str(args.output_prefix) + ".features.json"
        logging.info("Saving intermediate to %s", intermediate_output)
        json.dump(genome_features, open(intermediate_output, "w", encoding="utf-8"))
    if args.output_prefix is not None and predictions is not None:
        output = str(args.output_prefix) + ".predictions.tsv"
        logging.info("Saving output to: %s", output)
        save_predictions(predictions, output_tsv=output)

    print(pd.DataFrame(predictions).T)
