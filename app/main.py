from app.models.numeric_model import Model as NumberModel
from app.models.ml_model import Model as MlModel
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
)

from app.utils.constants import data_path, metric_names

from tqdm import tqdm


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
    parser = argparse.ArgumentParser(description="Hybrid model")

    subparsers = parser.add_subparsers(help="sub-command help", dest="command")

    parser_predict = subparsers.add_parser("predict", help="predict help")

    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate help")

    args = parser.parse_args()

    if args.command == "predict":
        predict(args)
    elif args.command == "evaluate":
        evaluate(args)


def predict(args):
    X_train = pd.read_csv(f"{data_path}/X_train.csv")
    X_test = pd.read_csv(f"{data_path}/X_test.csv")

    print(
        f"Number of rows used to predict model. Train: {len(X_train)}. Test: {len(X_test)}"
    )

    for [data_type, X] in tqdm(
        [
            ["train", X_train],
            ["test", X_test],
        ]
    ):
        model = Model()

        predictions = model.predict(X)

        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_csv(
            f"app/models/predictions/hybrid_model/{data_type}.csv", index=False
        )


def evaluate(args):
    y_train = pd.read_csv(f"{data_path}/y_train.csv")
    predictions_train = pd.read_csv(f"app/models/predictions/hybrid_model/train.csv")
    y_test = pd.read_csv(f"{data_path}/y_test.csv")
    predictions_test = pd.read_csv(f"app/models/predictions/hybrid_model/test.csv")

    print(
        f"Number of rows used to evaluate model. Train: {len(y_train)}. Test: {len(y_test)}"
    )

    for [data_type, predictions, y_true] in [
        ["train", predictions_train, y_train],
        ["test", predictions_test, y_test],
    ]:
        res = []

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

        df_res = pd.DataFrame(res, index=["Hybrid model"], columns=metric_names)
        df_res.to_excel(f"app/models/tables/hybrid_model/{data_type}.xlsx")

        ConfusionMatrixDisplay.from_predictions(y_true, predictions)
        plt.savefig(
            f"app/models/figures/hybrid_model/{data_type}_confusion_matrix.png",
            bbox_inches="tight",
        )
        RocCurveDisplay.from_predictions(y_true, predictions)
        plt.savefig(
            f"app/models/figures/hybrid_model/{data_type}_roc_curve.png",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
