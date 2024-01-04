"""Functions for training models"""
import json
from pathlib import Path

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..helpers import rename_condition_to_variable
from ..taxonomy import (
    BalanceTaxa,
    TaxonomyGTDB,
)
from .make_holdout_sets import balance_but_keep_extremes


rng = np.random.default_rng(0)
ROOT_DIR = str(Path(__file__).resolve().parent.parent)


def yield_cv_sets(cv_sets):
    """Generator for sklearn to handle CV sets"""
    for training_indices, validation_indices in cv_sets:
        yield training_indices, validation_indices


# load_cv_sets


def predict_training_and_cv(X_train, y_train, pipeline, cv, method="predict"):
    """Wrapper to provide consistent output

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


def score_regression(y_true, y_pred):
    statistics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "corr": np.corrcoef(y_true, y_pred)[1, 0],
    }
    return statistics


def score_classification(y_true, y_pred_probs, p_threshold=0.5):
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


### FINAL


def load_instructions(instructions_filename: str):
    """Load pipeline and features stored in instruction file"""
    with open(instructions_filename) as fh:
        instructions = json.loads(fh.read())
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
        elif condition == "oxygen":
            target = condition
            print(f"Training {target}")
            pipeline = joblib.load(pipeline_filename)
            pipeline = train_model(pipeline, training_df, features, target)


if __name__ == "__main__":
    train_models(
        training_data_filename=f"{ROOT_DIR}/data/training_data/training_data_20231203.tsv",
        instructions_filename=f"{ROOT_DIR}/models/instructions.json",
    )
