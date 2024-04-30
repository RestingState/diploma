from app.models.numeric_model import Model as NumberModel
from app.models.ml_model import Model as MlModel
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse

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


class Model:
    def __init__(self):
        self.numberModel = NumberModel()
        self.mlModel = MlModel()

    def predict(self, jobs):
        numeric_predictions = self.numberModel.predict(jobs)
        ml_predictions = self.mlModel.predict(jobs)

        predictions = numeric_predictions & ml_predictions

        return predictions


def main():
    parser = argparse.ArgumentParser(description="main")

    parser.add_argument("-n", "--number-of-rows", type=int)

    args = parser.parse_args()

    X_test = pd.read_csv(f"{data_path}/X_test.csv")
    y_test = pd.read_csv(f"{data_path}/y_test.csv")

    if args.number_of_rows is not None:
        X_test = X_test.iloc[: args.number_of_rows]
        y_test = y_test.iloc[: args.number_of_rows]

    print(f"Number of rows used to evaluate model: {len(X_test)}")

    model = Model()

    st = time.time()
    predictions = model.predict(X_test)
    et = time.time()

    print(f"Accuracy score: {accuracy_score(y_test, predictions)}")
    print(f"Precision score: {precision_score(y_test, predictions)}")
    print(f"Recall score: {recall_score(y_test, predictions)}")
    print(f"F1 score: {f1_score(y_test, predictions)}")
    print(f"MCC score: {matthews_corrcoef(y_test, predictions)}")
    print(f"Cohen's kappa score: {cohen_kappa_score(y_test, predictions)}")

    elapsed_time = et - st
    print(f"Execution time: {elapsed_time} seconds")

    print(classification_report(y_test, predictions))

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.show()


if __name__ == "__main__":
    main()
