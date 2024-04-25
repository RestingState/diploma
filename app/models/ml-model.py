from transformers import pipeline
import argparse
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import evaluate
import time

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


class ModelNames:
    distilbertBaseUncased = "distilbert-base-uncased"
    bertBaseUncased = "bert-base-uncased"


data_path = "app/data/jobs-main-data.csv"
hub_name = "DenysZakharkevych"
max_length = 512


class ModelColumns:
    budget = "budget"
    hourlyRangeMin = "hourlyRangeMin"
    isHourlyPayment = "isHourlyPayment"
    country = "country"
    category = "category"
    workload = "workload"
    duration = "duration"
    clientTotalCharge = "clientTotalCharge"
    clientTotalJobsPosted = "clientTotalJobsPosted"
    clientFeedbackScore = "clientFeedbackScore"
    clientPastHires = "clientPastHires"
    isPaymentMethodVerified = "isPaymentMethodVerified"
    skills = "skills"
    target = "target"


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
accuracy = evaluate.load("accuracy")


def main():
    parser = argparse.ArgumentParser(description="Ml-model")

    subparsers = parser.add_subparsers(help="sub-command help", dest="command")

    # evaluate subparser
    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate help")
    parser_evaluate.add_argument("--with-plots", action="store_true")

    parser.add_argument(
        "model_name",
        choices=[ModelNames.distilbertBaseUncased, ModelNames.bertBaseUncased],
    )

    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate(args)


def evaluate(args):
    X_test, y_test = prepare_test_dataset()

    model_name = args.model_name

    model = AutoModelForSequenceClassification.from_pretrained(
        f"{hub_name}/{model_name}"
    )

    classifier = pipeline("sentiment-analysis", model=f"{hub_name}/{model_name}")

    st = time.time()
    predictions = classifier(X_test)
    et = time.time()

    predictions = [model.config.label2id[x["label"]] for x in predictions]

    print(f"Accuracy score: {accuracy_score(y_test, predictions)}")
    print(f"Precision score: {precision_score(y_test, predictions)}")
    print(f"Recall score: {recall_score(y_test, predictions)}")
    print(f"F1 score: {f1_score(y_test, predictions)}")
    print(f"MCC score: {matthews_corrcoef(y_test, predictions)}")
    print(f"Cohen's kappa score: {cohen_kappa_score(y_test, predictions)}")

    print(f"Number of rows used to evaluate model: {len(X_test)}")

    elapsed_time = et - st
    print(f"Execution time: {elapsed_time} seconds")

    print(classification_report(y_test, predictions))

    if args.with_plots:
        ConfusionMatrixDisplay.from_predictions(y_test, predictions)
        plt.show()


def prepare_test_dataset():
    df = pd.read_csv(data_path)

    df = df[df["status"].isin(["prelead", "in-progress"]) == False]

    def transform_status_into_target(row):
        if row["status"] == "trashed":
            return 0
        else:
            return 1

    df[ModelColumns.target] = df.apply(transform_status_into_target, axis=1)
    df = df.drop(
        columns=[
            "id",
            "uid",
            "score",
            "createdAt",
            "status",
            "postedAt",
            "query",
        ]
    )

    df.dropna(subset=[ModelColumns.country, ModelColumns.duration], inplace=True)

    # reassign jobs which do not have verified method payment, but were taken
    df.loc[
        (
            (df[ModelColumns.isPaymentMethodVerified] == 0)
            & (df[ModelColumns.target] == 1)
        ),
        ModelColumns.target,
    ] = 0

    # reassign project jobs which have budget less than 5000, but were taken
    df.loc[
        (
            (df[ModelColumns.target] == 1)
            & (df[ModelColumns.budget] < 5000)
            & (df[ModelColumns.isHourlyPayment] == 0)
        ),
        ModelColumns.target,
    ] = 0

    X = df.drop(ModelColumns.target, axis=1).copy()
    X = X[["title", "description"]]

    y = df[ModelColumns.target].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    X_test = list(X_test["title"] + "\n" + X_test["description"])

    X_test = [tokenizer(x, truncation=True, max_length=max_length - 2) for x in X_test]

    X_test = [tokenizer.decode(x["input_ids"]) for x in X_test]

    return X_test[:20], y_test[:20]


if __name__ == "__main__":
    main()
