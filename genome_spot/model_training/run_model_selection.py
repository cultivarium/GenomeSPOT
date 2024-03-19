"""Script to measure the performance of various estimators and features. 

The scores can be accessed and reviewed to select the best-performing model 
and features for each target variable. Manual review is reecommended in case
two models have similar performance, but the user prefers a simpler model
(which is less likely to be overfit). The results are saved in the directory
specified in the `outdir` argument.
"""

import argparse
import json
import logging
from datetime import datetime
from typing import (
    List,
    Tuple,
)

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
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
from .train_models import predict_and_score


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%H:%M:%S")


CONDITIONS = ["oxygen", "temperature", "salinity", "ph"]
COMPARTMENTS = ["extracellular_soluble", "intracellular_soluble", "membrane"]
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
    "nt_C",
    "pur_pyr_transition_freq",
    "protein_coding_density",
    "mean_protein_length",
]

BASE_VARS_ALL = BASE_VARS_DERIVED_GENOME + BASE_VARS_DERIVED_PROTEIN + BASE_VARS_AAS + BASE_VARS_PIS


class ModelSelection:
    """Provides data for model and feature selection by training and scoring
    different combinations of models and features, using cross-validation at the specified rank.
    Models and features are predefined and loaded using functions of this class.

    Args:
        cv_rank: str, the taxonomic rank at which to perform cross-validation


    Example usage:
        ```python
        df = pd.read_csv(training_data_filename, index_col=0, sep="\t")
        selection = ModelSelection()
        for condition in CONDITIONS:
            selection.score_all_models_and_feature_sets(df, condition, path_to_holdouts, outdir)
        ```
    """

    def __init__(self, cv_rank="family"):
        self.named_feature_sets = self.generate_named_feature_sets()
        self.feature_set_names = [tup[0] for tup in self.named_feature_sets]

        self.cv_rank = cv_rank
        self.conditions = CONDITIONS

        # features before localization
        self.conditions = CONDITIONS
        self.base_vars_aas = BASE_VARS_AAS
        self.base_vars_pis = BASE_VARS_PIS
        self.base_vars_derived_genome = BASE_VARS_DERIVED_GENOME
        self.base_vars_derived_protein = BASE_VARS_DERIVED_PROTEIN
        self.base_vars_all = BASE_VARS_ALL

        # types of localization (aside from none)
        self.COMPARTMENTS = COMPARTMENTS

    def generate_named_feature_sets(self) -> List[Tuple[str, List[str]]]:
        """Uses global constants to create sets of features
        to be used in model selection

        Feature sets are defined by the type of feature and by the localization.
        The type of


        Localization of protein features is either none, by compartment
        (extracellular_soluble, intracellular_soluble, membrane)
        or by difference between extracellular and intracellular
        compartments.

        Returns:
            named_feature_sets: A list of tuples where the first item is a name of the
                feature set and the second item is a list of features in the set.
        """

        # whole-genome
        vars_aas = prepend_features(BASE_VARS_AAS, prefices=["all"])
        vars_pis = prepend_features(BASE_VARS_PIS, prefices=["all"])
        vars_derived = prepend_features(BASE_VARS_DERIVED_GENOME + BASE_VARS_DERIVED_PROTEIN, prefices=["all"])
        vars_all = vars_aas + vars_pis + vars_derived

        # split by compartments unless a DNA attribute
        vars_aas_by_compartment = prepend_features(BASE_VARS_AAS, prefices=COMPARTMENTS)
        vars_pis_by_compartment = prepend_features(BASE_VARS_PIS, prefices=COMPARTMENTS)
        vars_derived_by_compartment = prepend_features(
            BASE_VARS_DERIVED_PROTEIN, prefices=COMPARTMENTS
        ) + prepend_features(BASE_VARS_DERIVED_GENOME, prefices=["all"])
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

    def load_pipeline_for_condition(self, condition: str, features: list) -> list:
        """Loads a set of pipelines for a given condition and set of features."""
        if condition in ["temperature", "salinity", "ph"]:
            pipelines = self.load_regressors(features)
        elif condition == "oxygen":
            pipelines = self.load_classifiers(features)
        return [self._make_pipeline(pipeline) for pipeline in pipelines]

    def _make_pipeline(self, estimators: list):
        """Wrapper for sklearn make_pipeline function, which creates a
        pipeline from a list of estimators such as `[StandardScaler(), LogisticRegression()]`
        """
        pipeline = None
        try:
            pipeline = make_pipeline(*estimators)
        except:  # some estimators like Random Forest require a workaround
            pipeline = make_pipeline(estimators)
        try:  # extend maxiter
            if pipeline[-1].max_iter < 10000:
                pipeline[-1].max_iter = 10000
        except:
            pass
        return pipeline

    def load_regressors(self, features: List[str]) -> List:
        """Defines a set of sklearn pipelines for regression. All pipelines use
        standard scaling, feature selection*, and a regressor.

        *Feature selection (using SelectKBest with either mutual_info_regression or
        f_regression) is applied to estimators that lack internal feature selection.
        Other estimators (Lasso, ElasticNet, MPLRegressor) have internal feature selection.

        Args:
            features: list of str, features to be used in regression
        Returns:
            regressors: list of pipelines
        """
        scaler = StandardScaler()
        select_kbest = [
            SelectKBest(regr, k=min([20, len(features)])) for regr in (mutual_info_regression, f_regression)
        ]

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
                Lasso(
                    alpha=alpha,
                ),
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
            *linear,
            *ridge,
            *lasso,
            *mlp_regressor,
        ]
        return regressors

    def load_classifiers(self, features: List[str]) -> list:
        """Defines a set of sklearn pipelines for classification. All pipelines use
        standard scaling, feature selection*, and a classifier.

        *Feature selection (using SelectKBest with either mutual_info_regression or
        f_regression) is applied to estimators that lack internal feature selection.
        SVC has internal feature selection.

        Args:
            features: list of str, features to be used in classification
        """
        select_kbest = [
            SelectKBest(regr, k=min([20, len(features)])) for regr in (mutual_info_regression, f_regression)
        ]

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
            *gaussian_nb,
            *logistic,
            *svc,
        ]

        return classifiers

    def score_all_models_and_feature_sets(self, df: pd.DataFrame, condition: str, path_to_holdouts: str, outdir: str):
        """Score performance all models and feature sets for a given condition. Save the performance.

        Args:
            df: pd.DataFrame, the training data
            condition: str, the condition to be predicted (e.g. 'temperature')
            path_to_holdouts: str, the path to the directory containing holdout sets
        Returns:
            None
        """
        feature_sets = self.generate_named_feature_sets()
        for i, (set_name, features) in enumerate(feature_sets):
            pipelines = self.load_pipeline_for_condition(condition, features)
            for j, pipeline in enumerate(pipelines):
                logging.info("Predicting %s with features %i and pipeline %i", condition, i, j)
                target = rename_condition_to_variable(condition)
                prefix = f"{target}_features{i}_pipeline{j}"
                self.score_model_and_save(df, condition, features, set_name, pipeline, path_to_holdouts, outdir, prefix)

    def score_model_and_save(self, df, condition, features, set_name, pipeline, path_to_holdouts, outdir, prefix):
        """Score and save the performance of a model in cross-validation at the specified
        taxonomic rank."""
        df_train, _ = split_train_and_test_data(df, condition, path_to_holdouts=path_to_holdouts)
        cv_sets = load_cv_sets(condition, path_to_holdouts=path_to_holdouts, taxlevel=self.cv_rank)
        target = rename_condition_to_variable(condition)

        cv_score = predict_and_score(
            X_train=df_train[features],
            y_train=df_train[target],
            condition=condition,
            cv_sets=cv_sets,
            pipeline=pipeline,
        )
        results = self.save_performance(
            pipeline=pipeline,
            feature_set=set_name,
            features=features,
            target=target,
            prefix=prefix,
            outdir=outdir,
            validation_statistics=cv_score,
        )
        return cv_score

    def save_performance(self, pipeline, target, feature_set, features, outdir, prefix, validation_statistics={}):
        """Save the model performance to a file and save the model to a file."""
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
            "cv_rank": self.cv_rank,
        }

        joblib.dump(pipeline, model_file)
        json.dump(results, open(results_file, "w"))
        return results


def run_model_selection(training_data_filename: str, path_to_holdouts: str, outdir: str):
    """Measure performance of different model and features on the training dataset
    to provide data for selecting a model and features for each target variable.

    Args:
        training_data_filename: str, path to training data TSV file
        path_to_holdouts: str, path to directory to save holdout sets
        outdir: str, path to directory to save model selection results
    """
    logging.info("Loading training data from %s", training_data_filename)
    df = pd.read_csv(training_data_filename, index_col=0, sep="\t")
    selection = ModelSelection()
    for condition in CONDITIONS:
        logging.info("Scoring models and features for %s", condition)
        selection.score_all_models_and_feature_sets(df, condition, path_to_holdouts, outdir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make holdout sets for model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training_data_filename",
        type=str,
        required=True,
        help="Path to training data TSV file",
    )

    parser.add_argument(
        "--path_to_holdouts",
        type=str,
        required=True,
        help="Path to directory to save holdout sets",
    )
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")

    args = parser.parse_args()

    if str(args.outdir).endswith("/"):
        args.outdir = args.outdir[:-1]
    return args


if __name__ == "__main__":
    args = parse_args()
    run_model_selection(
        training_data_filename=args.training_data_filename,
        path_to_holdouts=args.path_to_holdouts,
        outdir=args.outdir,
    )
