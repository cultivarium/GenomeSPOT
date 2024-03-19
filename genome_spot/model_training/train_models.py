"""Functions for training models"""

import argparse
import json
import logging
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from ..helpers import (
    load_train_and_test_sets,
    load_training_data,
    rename_condition_to_variable,
)
from .make_holdout_sets import (
    make_cv_sets_by_phylogeny,
    yield_cv_sets,
)


logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", format="%(asctime)s %(levelname)s %(message)s")
rng = np.random.default_rng(0)


def load_instructions(instructions_filename: str) -> dict:
    """Loads and validates a dictionary with
    instructions for pipeline and features

    Args:
        instructions_filename (str): path to json file
    Returns:
        instructions (dict): dictionary with keys as conditions
    """
    with open(instructions_filename) as fh:
        instructions = json.loads(fh.read())
        for condition in instructions.keys():
            assert "pipeline_filename" in instructions[condition].keys()
            assert "features" in instructions[condition].keys()

    return instructions


def train_model(
    pipeline: Pipeline, df_train: pd.DataFrame, features: list, target: str, path_to_models: str, save: bool = True
) -> Pipeline:
    """Trains a model using a pipeline and saves it

    Args:
        pipeline (Pipeline): pipeline to fit
        df_train (pd.DataFrame): training data
        features (list): list of features to use
        target (str): name of target variable
        path_to_models (str): path to models directory
        save (bool): whether to save the model
    Returns:
        pipeline (Pipeline): fitted pipeline
    """
    X_train = df_train[features].values
    y_train = df_train[target]
    pipeline[-1].max_iter = 50000  # spare no expense!
    pipeline.fit(X_train, y_train)
    if save is True:
        joblib.dump(pipeline, f"{path_to_models}/{target}.joblib")
        save_data(target, features, genome_accessions=df_train.index.tolist(), path_to_models=path_to_models)
    return pipeline


def train_novelty_detection_model(
    df_train: pd.DataFrame, features: list, target: str, path_to_models: str, nu: float = 0.02, save: bool = True
) -> OneClassSVM:
    """Creates a novelty detection model using OneClassSVM.

    The parameter `nu` (0<=nu<=1) controls how much data is assigned as novel. For example,
    nu=0.02 means that 2% of the training dataset will be considered novel. Data not
    in the training set will include a higher proportion of novel data simply by
    statistical chance, but perhaps also due to biological reasons.

    Args:
        df_train (pd.DataFrame): training data
        features (list): list of features to use
        target (str): name of target variable
        path_to_models (str): path to models directory
        nu (float): proportion of training data to be considered novel
        save (bool): whether to save the model
    Returns:
        novelty_model (OneClassSVM): fitted novelty detection model
    """
    condition = target.split("_")[0]
    X_train = df_train[features].values
    novelty_model = OneClassSVM(nu=nu).fit(X_train)
    novelty_model.fit(X_train)
    if save is True:
        joblib.dump(novelty_model, f"{path_to_models}/novelty_{condition}.joblib")
    return novelty_model


def train_error_model(
    pipeline: Pipeline, df_train: pd.DataFrame, features: list, target: str, path_to_models: str, save: bool = True
) -> np.ndarray:
    """Creates an 'error model' which is the RMSE for each value
    predicted by model in cross-validation within a given interval.

    For example, for a predicted pH of 10 and an interval of 1 pH unit,
    the model provides the RMSE for predicted values between pH 9.5-10.5
    during cross-validation. This is an estimate of the error for the new
    prediction. This does not work for classifiers, such as the oxygen
    classifier. The model is later used by finding the RMSE of the cross-
    validation prediction closest to the prediction of interest.

    Args:
        pipeline (Pipeline): pipeline to fit
        df_train (pd.DataFrame): training data
        features (list): list of features to use
        target (str): name of target variable
        save (bool): whether to save the model
    Returns:
        error_model (np.ndarray): array of predicted values (col 1) and RMSE (col 2)
    """
    interval_dict = {
        "ph": 1,
        "salinity": 2,
        "temperature": 10,
    }
    condition = target.split("_")[0]
    X_train = df_train[features].values
    y_train = df_train[target]
    if condition == "oxygen":
        return None
    else:
        partition_rank = "family"
        cv_sets = make_cv_sets_by_phylogeny(genomes=df_train.index.tolist(), partition_rank=partition_rank, kfold=5)
        method = "predict"
        _, _, y_valid_pred = predict_training_and_cv(
            X_train, y_train, pipeline, cv=yield_cv_sets(cv_sets), method=method
        )
        error_model = rmse_by_value(y_train, y_valid_pred, interval=interval_dict[condition])
        if save is True:
            joblib.dump(error_model, f"{path_to_models}/error_{target}.joblib")
        return error_model


def rmse_by_value(y_true: np.ndarray, y_pred: np.ndarray, interval: float) -> np.ndarray:
    """Returns an array where col 1 is the value and col 2
    is the RMSE of values predicted to be y +/- interval

    Args:
        y_true (np.ndarray): true values
        y_pred (np.ndarray): predicted values
        interval (float): interval to compute RMSE overall
    Returns:
        error_arr (np.ndarray): array of predicted values (col 1) and RMSE (col 2)
    """
    rmse_arr = np.empty(y_true.shape)
    for i, y in enumerate(y_pred):
        mask = (y_pred > (y - interval / 2)) & (y_pred < (y + interval / 2))
        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        rmse_arr[i] = rmse

    error_arr = np.stack([y_pred, rmse_arr], axis=1)
    return error_arr


def predict_training_and_cv(
    X_train, y_train, pipeline, cv, method="predict"
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """Wrapper to provide consistent output

    Args:
        X_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        pipeline (sklearn.pipeline.Pipeline): pipeline to fit
        cv (generator): generator of CV sets
        method (str): 'predict' or 'predict_proba'
    Returns:
        - pipeline: estimators fit to all data
        - y_train_pred: values predicted by model fit to all data
        - y_valid_pred: values predicted in cross-validation
            by model not trained on those values
    """
    y_valid_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv, method=method)
    pipeline.fit(X_train, y_train)
    if method == "predict":
        y_train_pred = pipeline.predict(X_train)
    elif method == "predict_proba":
        y_train_pred = pipeline.predict_proba(X_train)
    return pipeline, y_train_pred, y_valid_pred


def predict_and_score(condition, X_train, y_train, cv_sets, pipeline):
    """Predict and score the performance of a pipeline on a given condition."""
    if condition in ["temperature", "salinity", "ph"]:
        pipeline, y_train_pred, y_valid_pred = predict_training_and_cv(
            X_train, y_train, pipeline=pipeline, cv=yield_cv_sets(cv_sets), method="predict"
        )
        validation_statistics = score_regression(y_train, y_valid_pred)
    elif condition == "oxygen":
        pipe, y_train_pred_probs, y_valid_pred_probs = predict_training_and_cv(
            X_train,
            y_train,
            pipeline=pipeline,
            cv=yield_cv_sets(cv_sets),
            method="predict_proba",
        )
        validation_statistics = score_classification(y_train.values, y_valid_pred_probs[:, 1])
    return validation_statistics


def score_regression(y_true, y_pred) -> dict:
    """Provides a dictionary of common statistics for regression models.

    Args:
        y_true (np.ndarray): true values
        y_pred (np.ndarray): predicted values
    Returns:
        statistics (dict): dictionary with keys 'rmse', 'r2', 'corr'
    """
    statistics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "corr": np.corrcoef(y_true, y_pred)[1, 0],
    }
    return statistics


def score_classification(y_true, y_pred_probs, p_threshold=0.5) -> dict:
    """Provides a dictionary of common statistics for binary classifier.

    Args:
        y_true (np.ndarray): true values
        y_pred_probs (np.ndarray): predicted probabilities for class 1
    Returns:
        statistics (dict): dictionary with keys 'f1', 'specificity', 'sensitivity'
    """
    y_pred = y_pred_probs > p_threshold
    conf = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf.ravel()
    specificity = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    f1 = f1_score(y_true, y_pred)
    statistics = {
        "f1": np.mean(f1),
        "specificity": np.mean(specificity),
        "sensitivity": np.mean(sensitivity),
    }
    return statistics


def save_data(target: str, features: list, genome_accessions: list, path_to_models: str) -> dict:
    """Saves instructions for each model

    Args:
        target (str): name of target variable
        features (list): list of features used in model
        genome_accessions (list): list of genome accessions used in model
        path_to_models (str): path to models directory

    Returns:
        record (dict): dictionary with keys 'target', 'features', 'genome_accessions'
    """
    record = {
        "target": target,
        "features": features,
        "genome_accessions": genome_accessions,
    }
    with open(f"{path_to_models}/{target}.instructions.json", "w") as fh:
        json.dump(record, fh)

    return record


def train_model_for_each_condition(
    condition: str,
    df: pd.DataFrame,
    instructions: dict,
    path_to_models: str,
    path_to_holdouts: str,
):
    """Trains models for each condition

    Models include:
    - Main regressor or classifier
    - Novelty detection model (2% of training data considered novel)
    - Error model (for regression only)

    Models are saved under the path_to_models directory.

    Args:
        condition (str): name of condition
        df (pd.DataFrame): training data
        instructions (dict): dictionary with instructions for each model
        path_to_models (str): path to models directory
        path_to_holdouts (str): path to directory with train and test sets for each condition
    Returns:
        None
    """
    logging.info("Loading data for %s", condition)
    pipeline_filename = instructions[condition]["pipeline_filename"]
    features = instructions[condition]["features"]
    target = rename_condition_to_variable(condition)
    condition_df = df[df[f"use_{condition}"] == True]
    logging.info("%s data available for %s genomes", condition, len(condition_df))

    train_set, test_set = load_train_and_test_sets(condition, path_to_holdouts)
    balanced_genomes = list(set(train_set).union(test_set))
    training_df = condition_df.loc[list(balanced_genomes)]
    logging.info("%s data available for %s genomes in train and test sets", condition, len(training_df))

    if condition in ["temperature", "ph", "salinity"]:
        for attr in ["optimum", "min", "max"]:
            logging.info("Training model for %s", target)
            target = f"{condition}_{attr}"
            pipeline = joblib.load(pipeline_filename)
            pipeline = train_model(pipeline, training_df, features, target, path_to_models=path_to_models, save=True)
            logging.info("Training model for %s novelty", target)
            novelty_model = train_novelty_detection_model(
                training_df, features, target, nu=0.02, path_to_models=path_to_models, save=True
            )
            logging.info("Training model for %s error", target)
            error_model = train_error_model(
                pipeline, training_df, features, target, path_to_models=path_to_models, save=True
            )
    elif condition == "oxygen":
        logging.info("Training model for %s", target)
        target = condition
        pipeline = joblib.load(pipeline_filename)
        pipeline = train_model(pipeline, training_df, features, target, path_to_models=path_to_models, save=True)
        logging.info("Training model for %s novelty", target)
        novelty_model = train_novelty_detection_model(
            training_df, features, target, nu=0.02, path_to_models=path_to_models, save=True
        )
        logging.info("Training model for %s error", target)
        error_model = train_error_model(
            pipeline, training_df, features, target, path_to_models=path_to_models, save=True
        )


def train_models(
    training_data_filename: str,
    path_to_models: str,
    path_to_holdouts: str,
):
    """Main function to train and save models for each condition

    Args:
        training_data_filename (str): path to training data
        path_to_models (str): path to models directory
        path_to_holdouts (str): path to directory with train and test sets for each condition
    Returns:
        None
    """
    df = load_training_data(training_data_filename)
    instructions = load_instructions(f"{path_to_models}/instructions.json")
    logging.info("Training models for conditions: %s", ", ".join(instructions.keys()))
    for condition in instructions.keys():
        train_model_for_each_condition(
            condition, df, instructions, path_to_models=path_to_models, path_to_holdouts=path_to_holdouts
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models for each condition based on instruction file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training_data_filename",
        type=str,
        required=True,
        help="Path to training data TSV file",
    )
    parser.add_argument(
        "--path_to_models",
        type=str,
        required=True,
        help="Path to directory to save models, containing instructions.json",
    )
    parser.add_argument(
        "--path_to_holdouts",
        type=str,
        required=True,
        help="Path to directory with train and test sets for each condition",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_models(
        training_data_filename=args.training_data_filename,
        path_to_models=args.path_to_models,
        path_to_holdouts=args.path_to_holdouts,
    )
