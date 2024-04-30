from transformers import pipeline
import argparse
import matplotlib.pyplot as plt
import numpy as np

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

from app.utils.constants import data_path


class ModelNames:
    distilbertBaseUncased = "distilbert-base-uncased"
    bertBaseUncased = "bert-base-uncased"


hub_name = "DenysZakharkevych"
max_length = 512
num_samples = 20


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
accuracy = evaluate.load("accuracy")


class Model:
    def __init__(self, model_name=ModelNames.distilbertBaseUncased):
        self.classifier = pipeline(
            "sentiment-analysis", model=f"{hub_name}/{model_name}"
        )

    def predict(self, jobs):
        jobs = list(jobs["title"] + "\n" + jobs["description"])

        jobs = [tokenizer(x, truncation=True, max_length=max_length - 2) for x in jobs]

        jobs = [tokenizer.decode(x["input_ids"]) for x in jobs]

        predictions = self.classifier(jobs)

        predictions = [label2id[x["label"]] for x in predictions]

        return np.array(predictions)


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

    classifier = pipeline("sentiment-analysis", model=f"{hub_name}/{model_name}")

    st = time.time()
    predictions = classifier(X_test)
    et = time.time()

    predictions = [label2id[x["label"]] for x in predictions]

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
    X_test = pd.read_csv(f"{data_path}/X_test.csv")
    y_test = pd.read_csv(f"{data_path}/y_test.csv")

    X_test = list(X_test["title"] + "\n" + X_test["description"])

    X_test = [tokenizer(x, truncation=True, max_length=max_length - 2) for x in X_test]

    X_test = [tokenizer.decode(x["input_ids"]) for x in X_test]

    return X_test[:num_samples], y_test[:num_samples]


if __name__ == "__main__":
    main()
