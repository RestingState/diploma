import pandas as pd
import numpy as np
import plotly.graph_objects as go


def main():
    df_train = pd.read_excel(
        "app/models/tables/numeric_model/train_close.xlsx", index_col=0
    )
    df_test = pd.read_excel(
        "app/models/tables/numeric_model/test_close.xlsx", index_col=0
    )

    model_names = df_train.index.values

    y_train_acc_scores = df_train["Accuracy"]
    y_test_acc_scores = df_test["Accuracy"]

    y_train_f1_scores = df_train["F1 score"]
    y_test_f1_scores = df_test["F1 score"]

    fig = go.Figure(
        data=[
            go.Bar(name="Train", x=model_names, y=y_train_acc_scores),
            go.Bar(name="Test", x=model_names, y=y_test_acc_scores),
        ]
    )

    fig.update_layout(
        barmode="group",
        title=f"Порівняння accuracy(точність) метрики",
        yaxis_range=[0, 1],
        font={"size": 18},
    )
    fig.write_image(
        "app/models/figures/numeric_model/accuracy.png",
        scale=4,
        width=1000,
        height=500,
    )

    fig = go.Figure(
        data=[
            go.Bar(name="Train", x=model_names, y=y_train_f1_scores),
            go.Bar(name="Test", x=model_names, y=y_test_f1_scores),
        ]
    )

    fig.update_layout(
        barmode="group",
        title=f"Порівняння F1 score метрики",
        yaxis_range=[0, 1],
        font={"size": 18},
    )
    fig.write_image(
        "app/models/figures/numeric_model/f1.png",
        scale=4,
        width=1000,
        height=500,
    )


if __name__ == "__main__":
    main()
