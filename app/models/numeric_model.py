import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from joblib import dump, load
import time
from xgboost import XGBClassifier
from app.utils.constants import data_path, ModelColumns


class ModelNames:
    svc = "svc"
    rfc = "rfc"
    knn = "knn"
    xgb = "xgb"


class Model:
    def __init__(self, model_name=ModelNames.svc):
        self.pipeline = load_model(f"{model_name}_pipeline.joblib")

    def predict(self, jobs):
        predictions = self.pipeline.predict(jobs)

        return predictions


def main():
    parser = argparse.ArgumentParser(description="Jobs scoring")

    subparsers = parser.add_subparsers(help="sub-command help", dest="command")

    # train subparser
    parser_train = subparsers.add_parser("train", help="train help")
    parser_train.add_argument("-n", "--number-of-rows", type=int)

    # evaluate subparser
    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate help")
    parser_evaluate.add_argument("--with-plots", action="store_true")

    parser.add_argument(
        "model_name",
        choices=[ModelNames.svc, ModelNames.rfc, ModelNames.knn, ModelNames.xgb],
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)


def train(args):
    X_train = pd.read_csv(f"{data_path}/X_train.csv")
    y_train = pd.read_csv(f"{data_path}/y_train.csv")

    if args.number_of_rows is not None:
        X_train = X_train.head(args.number_of_rows)
        y_train = y_train.head(args.number_of_rows)

    preprocessing_pipeline = get_preprocessing_pipeline()

    model_name = args.model_name
    if model_name == ModelNames.svc:
        model = SVC(random_state=42, C=100, gamma=0.001, kernel="rbf")
    elif model_name == ModelNames.rfc:
        model = RandomForestClassifier()
    elif model_name == ModelNames.knn:
        model = KNeighborsClassifier()
    elif model_name == ModelNames.xgb:
        model = XGBClassifier(objective="binary:logistic")

    pipeline = make_pipeline(preprocessing_pipeline, model)

    st = time.time()
    pipeline.fit(X_train, y_train)
    et = time.time()

    print(f"Number of rows used to train model: {len(X_train)}")

    elapsed_time = et - st
    print(f"Execution time: {elapsed_time} seconds")

    save_model(pipeline, f"{model_name}_pipeline.joblib")

    print(f"{model_name} model was successfully trained")


def evaluate(args):
    X_test = pd.read_csv(f"{data_path}/X_test.csv")
    y_test = pd.read_csv(f"{data_path}/y_test.csv")

    pipeline = load_model(f"{args.model_name}_pipeline.joblib")

    st = time.time()
    predictions = pipeline.predict(X_test)
    et = time.time()

    print(f"Accuracy score: {accuracy_score(y_test, predictions)}")
    print(f"Precision score: {precision_score(y_test, predictions, pos_label=0)}")
    print(f"Recall score: {recall_score(y_test, predictions, pos_label=0)}")
    print(f"F1 score: {f1_score(y_test, predictions, pos_label=0)}")
    print(f"MCC score: {matthews_corrcoef(y_test, predictions)}")
    print(f"Cohen's kappa score: {cohen_kappa_score(y_test, predictions)}")

    print(f"Number of rows used to evaluate model: {len(X_test)}")

    elapsed_time = et - st
    print(f"Execution time: {elapsed_time} seconds")

    print(classification_report(y_test, predictions))

    if args.with_plots:
        ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
        plt.show()


def get_preprocessing_pipeline():
    impute_numeric = ColumnTransformer(
        [
            (
                "impute_constant",
                SimpleImputer(strategy="constant", fill_value=0),
                [ModelColumns.budget, ModelColumns.hourlyRangeMin],
            ),
            (
                "impute_median",
                SimpleImputer(strategy="median"),
                [
                    ModelColumns.clientTotalCharge,
                    ModelColumns.clientTotalJobsPosted,
                    ModelColumns.clientFeedbackScore,
                    ModelColumns.clientPastHires,
                ],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    numeric_pipeline = make_pipeline(impute_numeric, StandardScaler())
    numeric_columns = [
        ModelColumns.budget,
        ModelColumns.hourlyRangeMin,
        ModelColumns.clientTotalCharge,
        ModelColumns.clientTotalJobsPosted,
        ModelColumns.clientFeedbackScore,
        ModelColumns.clientPastHires,
        ModelColumns.isHourlyPayment,
        ModelColumns.isPaymentMethodVerified,
    ]

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="infrequent_if_exist"),
    )
    categorical_columns = [
        ModelColumns.category,
        ModelColumns.workload,
        ModelColumns.duration,
    ]

    country_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=50),
    )

    skills_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="NaN"),
        ColumnTransformer(
            [("skill_vectorizer", CountVectorizer(binary=True, min_df=0.001), 0)]
        ),
    )

    column_trans = ColumnTransformer(
        [
            ("numeric_pipeline", numeric_pipeline, numeric_columns),
            ("categorical_pipeline", categorical_pipeline, categorical_columns),
            ("country_pipeline", country_pipeline, [ModelColumns.country]),
            ("skills_pipeline", skills_pipeline, [ModelColumns.skills]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return column_trans


def save_model(model, filename):
    dump(model, f"app/models/{filename}")


def load_model(filename) -> Pipeline:
    return load(f"app/models/{filename}")


if __name__ == "__main__":
    main()
