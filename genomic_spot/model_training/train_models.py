"""Functions for training models"""
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    ShuffleSplit,
    cross_val_predict,
)
from sklearn.pipeline import (
    Pipeline,
    make_pipeline,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ..helpers import rename_condition_to_variable
from ..taxonomy import (
    BalanceTaxa,
    TaxonomyGTDB,
)
from .make_holdout_sets import (
    balance_but_keep_extremes,
    make_cv_sets_by_phylogeny,
)


rng = np.random.default_rng(0)
ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)


def yield_cv_sets(cv_sets):
    """Generator for sklearn to handle CV sets"""
    for training_indices, validation_indices in cv_sets:
        yield training_indices, validation_indices


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


def load_training_data(training_data_filename) -> pd.DataFrame:
    training_df = pd.read_csv(training_data_filename, sep="\t", index_col=0)
    return training_df


def save_data(target, features, genome_accessions):
    record = {
        "target": target,
        "features": features,
        "genome_accessions": genome_accessions,
    }
    with open(f"models/{target}.instructions.json", "w") as fh:
        json.dump(record, fh)


def train_model(pipeline, df_train, features, target, save: bool = True):
    condition = target.split("_")[0]
    X_train = df_train[features].values
    y_train = df_train[target]
    pipeline[-1].max_iter = 50000  # spare no expense!
    pipeline.fit(X_train, y_train)
    if save is True:
        joblib.dump(pipeline, f"{ROOT_DIR}/models/{target}.joblib")
        save_data(target, features, genome_accessions=df_train.index.tolist())
    return pipeline


def train_novelty_detection_model(df_train, features, target, nu=0.02, save: bool = True):
    """Creates a novelty detection model using OneClassSVM.

    The parameter `nu` controls how much data is assigned as novel. For example,
    nu=0.02 means that 2% of the training dataset will be considered novel. Data not
    in the training set will include a higher proportion of novel data simply by
    statistical chance, but perhaps also due to biological reasons.
    """
    condition = target.split("_")[0]
    X_train = df_train[features].values
    clf = OneClassSVM(nu=nu).fit(X_train)
    clf.fit(X_train)
    if save is True:
        joblib.dump(clf, f"{ROOT_DIR}/models/novelty_{condition}.joblib")
    return clf


def train_error_model(pipeline, df_train, features, target, save: bool = True):
    """Creates an 'error model' which is the RMSE for each value
    predicted by model in cross-validation within a given interval

    For example, for a predicted pH of 10 and an interval of 1 pH unit,
    the model provides the RMSE for predicted values between pH 9.5-10.5
    during cross-validation. This is an estimate of the error for the new
    prediction. This does not work for classifiers, such as the oxygen
    classifier.
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
            joblib.dump(error_model, f"{ROOT_DIR}/models/error_{target}.joblib")

        return error_model


def rmse_by_value(y_true, y_pred, interval) -> np.ndarray:
    """Returns an array where col 1 is the value and col 2
    is the RMSE of values predicted to be y +/- interval
    """
    rmse_arr = np.empty(y_true.shape)
    for i, y in enumerate(y_pred):
        mask = (y_pred > (y - interval)) & (y_pred < (y + interval))
        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        rmse_arr[i] = rmse

    error_arr = np.stack([y_pred, rmse_arr], axis=1)
    return error_arr


def train_models(
    training_data_filename: str,
    instructions_filename: str,
):
    taxonomy = TaxonomyGTDB()
    balancer = BalanceTaxa(taxonomy=taxonomy)
    df = load_training_data(training_data_filename)
    print(df.shape)

    instructions = load_instructions(instructions_filename)
    print(instructions.keys())
    for condition in instructions.keys():
        # load
        pipeline_filename = instructions[condition]["pipeline_filename"]
        features = instructions[condition]["features"]
        target = rename_condition_to_variable(condition)
        print(target, condition)
        # subset to desired data
        condition_df = df[df[f"use_{condition}"] == True]
        print(condition_df.shape)
        balanced_genomes = balance_but_keep_extremes(
            df_data=condition_df,
            target=target,
            genomes_for_use=condition_df.index.drop_duplicates().tolist(),
            balancer=balancer,
        )
        training_df = condition_df.loc[list(balanced_genomes)]
        print(training_df.shape)
        # train
        if condition in ["temperature", "ph", "salinity"]:
            for attr in ["optimum", "min", "max"]:
                target = f"{condition}_{attr}"
                print(f"Training {target}")
                pipeline = joblib.load(pipeline_filename)
                pipeline = train_model(pipeline, training_df, features, target)
                novelty_model = train_novelty_detection_model(training_df, features, target, nu=0.02)
                error_model = train_error_model(pipeline, training_df, features, target, save=True)
        elif condition == "oxygen":
            target = condition
            print(f"Training {target}")
            pipeline = joblib.load(pipeline_filename)
            pipeline = train_model(pipeline, training_df, features, target)
            novelty_model = train_novelty_detection_model(training_df, features, target, nu=0.02)
            error_model = train_error_model(pipeline, training_df, features, target, save=True)


if __name__ == "__main__":
    train_models(
        training_data_filename=f"{ROOT_DIR}/data/training_data/training_data_20231203.tsv",
        instructions_filename=f"{ROOT_DIR}/models/instructions.json",
    )
