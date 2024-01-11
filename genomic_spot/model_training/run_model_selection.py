"""Script to run various estimators and features and report the performance of each"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    f_regression,
    mutual_info_regression,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..helpers import (
    load_cv_sets,
    prepend_features,
    rename_condition_to_variable,
    split_train_and_test_data,
)
from .make_holdout_sets import yield_cv_sets
from .train_models import (
    predict_training_and_cv,
    score_classification,
    score_regression,
)


# def rename_condition_to_variable(condition, attribute="optimum"):
#     """Commonly used to turn a condition e.g. 'temperature'
#     to a variable e.g. 'temperature_optimum'"""
#     if condition == "oxygen":
#         return "oxygen"
#     else:
#         return condition + "_" + attribute


# def prepend_features(features, prefices):
#     """useful for assigning localization to features"""
#     return [f"{prefix}_{feature}" for prefix in prefices for feature in features]


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
ROOT_DIR = str(Path(__file__).resolve().parent.parent)


TRAINING_DATA_TSV = f"{ROOT_DIR}/data/training_data/training_data_20231203.tsv"
PATH_TO_HOLDOUTS = f"{ROOT_DIR}/data/holdouts"


CONDITIONS = ["oxygen", "temperature", "salinity", "ph"]

# Exclude genome length-y things so MAGs can be predicted
# Types:
# - Genome attributes
# - Protein attributes
#   - Mean pI, zC, nH2O, GRAVY
#   - AA frequencies
#   - pI distributions
#   - Special: thermostable freq, R_RK frequency

BASE_VARS_AAS = [
    "aa_A",
    "aa_C",
    "aa_D",
    "aa_E",
    "aa_F",
    "aa_G",
    "aa_H",
    "aa_I",
    "aa_K",
    "aa_L",
    "aa_M",
    "aa_N",
    "aa_P",
    "aa_Q",
    "aa_R",
    "aa_S",
    "aa_T",
    "aa_V",
    "aa_W",
    "aa_Y",
]

BASE_VARS_PIS = [
    # "pis_3_4", # always 0
    "pis_4_5",
    "pis_5_6",
    "pis_6_7",
    "pis_7_8",
    "pis_8_9",
    "pis_9_10",
    "pis_10_11",
    "pis_11_12",
]

BASE_VARS_DERIVED_PROTEIN = [
    "mean_gravy",
    "mean_nh2o",
    "mean_pi",
    "mean_zc",
    "proportion_R_RK",
    "mean_thermostable_freq",
]

BASE_VARS_DERIVED_GENOME = [
    #'nt_length',
    #'total_proteins'
    #'total_protein_length',
    "nt_C",
    "pur_pyr_transition_freq",
    "protein_coding_density",
    "mean_protein_length",
]

BASE_VARS_ALL = BASE_VARS_DERIVED_GENOME + BASE_VARS_DERIVED_PROTEIN + BASE_VARS_AAS + BASE_VARS_PIS


def generate_named_feature_sets():
    """
    # by compartment, amino acids and pis
    # by compartment DIFF, amino acids and pis"""
    compartments = ["extracellular_soluble", "intracellular_soluble", "membrane"]

    # whole-genome
    vars_aas = prepend_features(BASE_VARS_AAS, prefices=["all"])
    vars_pis = prepend_features(BASE_VARS_PIS, prefices=["all"])
    vars_derived = prepend_features(BASE_VARS_DERIVED_GENOME + BASE_VARS_DERIVED_PROTEIN, prefices=["all"])
    vars_all = vars_aas + vars_pis + vars_derived

    # split by compartments unless a DNA attribute
    vars_aas_by_compartment = prepend_features(BASE_VARS_AAS, prefices=compartments)
    vars_pis_by_compartment = prepend_features(BASE_VARS_PIS, prefices=compartments)
    vars_derived_by_compartment = prepend_features(BASE_VARS_DERIVED_PROTEIN, prefices=compartments) + prepend_features(
        BASE_VARS_DERIVED_GENOME, prefices=["all"]
    )
    vars_all_by_compartment = vars_aas_by_compartment + vars_pis_by_compartment + vars_derived_by_compartment

    # difference between outside and in unless a DNA attribute
    vars_aas_diff_compartment = prepend_features(BASE_VARS_AAS, prefices=["diff_extra_intra"])
    vars_pis_diff_compartment = prepend_features(BASE_VARS_PIS, prefices=["diff_extra_intra"])
    vars_derived_diff_compartment = prepend_features(
        BASE_VARS_DERIVED_PROTEIN, prefices=["diff_extra_intra"]
    ) + prepend_features(BASE_VARS_DERIVED_GENOME, prefices=["all"])
    vars_all_diff_compartment = vars_aas_diff_compartment + vars_pis_by_compartment + vars_derived_diff_compartment

    named_feature_sets = [
        ("aas", vars_aas),
        ("aas_by_compartment", vars_aas_by_compartment),
        ("aas_diff_compartment", vars_aas_diff_compartment),
        ("pis", vars_pis),
        ("pis_by_compartment", vars_pis_by_compartment),
        ("pis_diff_compartment", vars_pis_diff_compartment),
        ("derived", vars_derived),
        ("derived_by_compartment", vars_derived_by_compartment),
        ("derived_diff_compartment", vars_derived_diff_compartment),
        ("all", vars_all),
        ("all_by_compartment", vars_all_by_compartment),
        ("all_diff_compartment", vars_all_diff_compartment),
    ]

    return named_feature_sets


def load_regressors(features):
    scaler = StandardScaler()
    select_kbest = [SelectKBest(regr, k=min([20, len(features)])) for regr in (mutual_info_regression, f_regression)]
    # select_lasso = [SelectFromModel(Lasso(alpha=alpha, max_iter=10_000), prefit=False) for alpha in (1e-3, 1e-1)]

    linear = [
        (
            scaler,
            selector,
            LinearRegression(fit_intercept=True),
        )
        for selector in select_kbest
    ]

    ridge = [
        (
            scaler,
            selector,
            Ridge(alpha=alpha),
        )
        for alpha in np.logspace(-3, 3, 7)
        for selector in select_kbest
    ]

    lasso = [
        (
            scaler,
            Lasso(alpha=alpha),
        )
        for alpha in np.logspace(-3, 0, 4)
    ]

    elasticnet = [
        (scaler, ElasticNet(l1_ratio=l1_ratio, alpha=alpha))
        for l1_ratio in [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
        for alpha in np.logspace(-3, 0, 4)
    ]

    mlp_regressor = [
        (
            scaler,
            MLPRegressor(hidden_layer_sizes=(int(len(features) * 1 / 2), 2), random_state=1, max_iter=5000),
        )
    ]

    regressors = [
        # *linear[0:2],  # test
        *linear,
        *ridge,
        *lasso,
        # *elasticnet,
        *mlp_regressor,
    ]
    return regressors


def load_classifiers(features):
    select_kbest = [SelectKBest(regr, k=min([20, len(features)])) for regr in (mutual_info_regression, f_regression)]

    gaussian_nb = [(StandardScaler(), selector, GaussianNB()) for selector in select_kbest]

    logistic = [
        (
            StandardScaler(),
            selector,
            LogisticRegression(
                random_state=0, max_iter=1000, class_weight="balanced", C=c, penalty=penalty, solver=solver
            ),
        )
        for selector in select_kbest
        for c in [1.0, 10.0, 100.0]
        for penalty, solver in zip(["l2", "l1"], ["lbfgs", "liblinear"])
    ]
    svc = [
        (
            StandardScaler(),
            SVC(class_weight="balanced", random_state=0, probability=True, kernel="linear", C=c),
        )
        for c in [1.0, 10.0, 100.0]
    ]
    classifiers = [
        # *logistic[0:2],  # test
        *gaussian_nb,
        *logistic,
        *svc,
    ]

    return classifiers


def save_information(pipeline, target, feature_set, features, outdir, prefix, validation_statistics={}):
    model_type = pipeline.get_params()["steps"][-1][0]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_file = f"{outdir}/{prefix}-{model_type}.json"
    model_file = f"{outdir}/{prefix}-{model_type}.joblib"
    results = {
        "prefix": prefix,
        "timestamp": timestamp,
        "results_file": results_file,
        "model_file": model_file,
        "target": target,
        "features": features,
        "model_type": model_type,
        "feature_set": feature_set,
        "performance": validation_statistics,
    }

    joblib.dump(pipeline, model_file)
    json.dump(results, open(results_file, "w"))
    return results


def _make_pipeline(estimators):
    # pipeline = None
    try:
        pipeline = make_pipeline(*estimators)
    except:
        pipeline = make_pipeline(estimators)
    return pipeline


def load_pipeline_for_condition(condition, features):
    if condition in ["temperature", "salinity", "ph"]:
        pipelines = load_regressors(features)
    elif condition == "oxygen":
        pipelines = load_classifiers(features)
    return [_make_pipeline(pipeline) for pipeline in pipelines]


def predict_and_score(condition, X_train, y_train, cv_sets, pipeline):
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


def run_model_selection():
    df = pd.read_csv(TRAINING_DATA_TSV, index_col=0, sep="\t")
    for condition in CONDITIONS:
        logging.info(condition)
        target = rename_condition_to_variable(condition)
        df_train, _ = split_train_and_test_data(df, condition, path_to_holdouts=PATH_TO_HOLDOUTS)
        cv_sets = load_cv_sets(condition, path_to_holdouts=PATH_TO_HOLDOUTS, taxlevel="family")
        feature_sets = generate_named_feature_sets()
        for i, (set_name, features) in enumerate(feature_sets):
            pipelines = load_pipeline_for_condition(condition, features)
            for j, pipeline in enumerate(pipelines):
                try:
                    if pipeline[-1].max_iter < 10000:
                        pipeline[-1].max_iter = 10000
                except:
                    pass
                model_type = pipeline.get_params()["steps"][-1][0]
                logging.info(
                    "CV for condition %s, features %i (%s), pipeline %i (%s)",
                    condition,
                    i,
                    set_name,
                    j,
                    model_type,
                )
                cv_score = predict_and_score(
                    X_train=df_train[features],
                    y_train=df_train[target],
                    condition=condition,
                    cv_sets=cv_sets,
                    pipeline=pipeline,
                )
                save_information(
                    pipeline=pipeline,
                    feature_set=set_name,
                    features=features,
                    target=target,
                    prefix=f"{target}_features{i}_pipeline{j}",
                    outdir=f"{ROOT_DIR}/data/model_selection/",
                    validation_statistics=cv_score,
                )


if __name__ == "__main__":
    run_model_selection()
