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
    RocCurveDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from joblib import dump, load
import time
from xgboost import XGBClassifier
from app.utils.constants import data_path, ModelColumns
from scipy.stats import uniform, randint
from tqdm import tqdm

# font = {"weight": "bold", "size": 20}
# plt.rc("font", **font)

# SMALL_SIZE = 14
# MEDIUM_SIZE = 16
# BIGGER_SIZE = 18

# plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
# plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
# plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
# plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
# plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
# plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.subplots_adjust(top=1.0)
# plt.subplots_adjust(wspace=0.0, hspace=0, right=0.7)


class ModelNames:
    svc = "svc"
    rfc = "rfc"
    knn = "knn"
    xgb = "xgb"


model_names = [ModelNames.svc, ModelNames.rfc, ModelNames.knn, ModelNames.xgb]


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
    parser_train.add_argument("--with-tuning", action="store_true")
    parser_train.add_argument(
        "model_name",
        choices=model_names,
    )

    # evaluate subparser
    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate help")
    parser_evaluate.add_argument("--with-plots", action="store_true")
    parser_evaluate.add_argument(
        "model_name",
        choices=model_names,
    )

    # evaluate all subparser
    parser_evaluate_all = subparsers.add_parser(
        "evaluate_all", help="evaluate all help"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "evaluate_all":
        evaluate_all(args)


def train(args):
    X_train = pd.read_csv(f"{data_path}/X_train.csv")
    y_train = pd.read_csv(f"{data_path}/y_train.csv")

    X_train = X_train.drop(columns=[ModelColumns.title, ModelColumns.description])
    y_train = y_train[ModelColumns.target].to_numpy()

    if args.number_of_rows is not None:
        X_train = X_train.head(args.number_of_rows)
        y_train = y_train.head(args.number_of_rows)

    print(f"Number of rows used to train model: {len(X_train)}")

    preprocessing_pipeline = get_preprocessing_pipeline()

    model_name = args.model_name
    if model_name == ModelNames.svc:
        model = SVC(random_state=42)
        distributions = {
            "svc__C": uniform(0.1, 100),
            "svc__gamma": uniform(0.001, 1),
            "svc__kernel": ["rbf", "poly"],
            "svc__degree": randint(2, 5),
        }
    elif model_name == ModelNames.rfc:
        model = RandomForestClassifier()
    elif model_name == ModelNames.knn:
        model = KNeighborsClassifier()
    elif model_name == ModelNames.xgb:
        model = XGBClassifier(objective="binary:logistic")

    pipeline = make_pipeline(preprocessing_pipeline, model)

    st = time.time()

    if args.with_tuning:
        clf = RandomizedSearchCV(
            pipeline,
            distributions,
            random_state=42,
            verbose=2,
            cv=3,
            n_iter=5,
            n_jobs=2,
            scoring="f1",
        )

        clf.fit(X_train, y_train)

        save_model(clf.best_estimator_, f"{model_name}_pipeline.joblib")
    else:
        pipeline.fit(X_train, y_train)

        save_model(pipeline, f"{model_name}_pipeline.joblib")

    et = time.time()

    elapsed_time = et - st
    print(f"Execution time: {elapsed_time} seconds")

    print(f"{model_name} model was successfully trained")


def evaluate(args):
    X_test = pd.read_csv(f"{data_path}/X_test.csv")
    y_test = pd.read_csv(f"{data_path}/y_test.csv")

    model_name = args.model_name

    model = Model(model_name=model_name)

    print(f"Model params: {model.pipeline.get_params()[model_name]}")

    st = time.time()
    predictions = model.predict(X_test)
    et = time.time()

    print(f"Accuracy score: {accuracy_score(y_test, predictions)}")
    print(f"Precision score: {precision_score(y_test, predictions)}")
    print(f"Recall score: {recall_score(y_test, predictions)}")
    print(f"F1 score: {f1_score(y_test, predictions)}")
    print(f"MCC score: {matthews_corrcoef(y_test, predictions)}")
    print(f"Cohen's kappa score: {cohen_kappa_score(y_test, predictions)}")
    print(f"AUC score: {roc_auc_score(y_test, predictions)}")

    print(f"Number of rows used to evaluate model: {len(X_test)}")

    elapsed_time = et - st
    print(f"Execution time: {elapsed_time} seconds")

    print(classification_report(y_test, predictions))

    if args.with_plots:
        ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        plt.savefig("app/models/figures/confusion_matrix.png", bbox_inches="tight")
        # plt.subplots_adjust(top=1.0)
        RocCurveDisplay.from_predictions(y_test, predictions)
        plt.savefig("app/models/figures/roc_curve.png", bbox_inches="tight")
        # plt.subplots_adjust(top=0.98, bottom=0.125)
        plt.show()


def evaluate_all(args):
    X_train = pd.read_csv(f"{data_path}/X_train.csv")
    y_train = pd.read_csv(f"{data_path}/y_train.csv")
    X_test = pd.read_csv(f"{data_path}/X_test.csv")
    y_test = pd.read_csv(f"{data_path}/y_test.csv")

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 score",
        "MCC",
        "Cohen’s kappa",
        "AUC",
    ]

    for [data_type, X, y_true] in [
        ["train", X_train, y_train],
        ["test", X_test, y_test],
    ]:
        res = []

        for model_name in tqdm(model_names):
            model = Model(model_name=model_name)

            predictions = model.predict(X)

            res.append(
                [
                    accuracy_score(y_true, predictions),
                    precision_score(y_true, predictions),
                    recall_score(y_true, predictions),
                    f1_score(y_true, predictions),
                    matthews_corrcoef(y_true, predictions),
                    cohen_kappa_score(y_true, predictions),
                    roc_auc_score(y_true, predictions),
                ]
            )

        df_res = pd.DataFrame(res, index=model_names, columns=metric_names)
        df_res.to_excel(f"app/models/tables/numeric_model/{data_type}.xlsx")


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
