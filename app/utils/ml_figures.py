import pandas as pd
import numpy as np
import plotly.graph_objects as go

model_names = ["DistilBERT", "ALBERT"]


def main():
    y_train_acc_scores = np.array([0.78, 0.74])
    y_test_acc_scores = np.array([0.73, 0.7])

    y_train_f1_scores = np.array([0.75, 0.71])
    y_test_f1_scores = np.array([0.71, 0.67])

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
        "app/models/figures/ml_model/accuracy.png",
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
        "app/models/figures/ml_model/f1.png",
        scale=4,
        width=1000,
        height=500,
    )


if __name__ == "__main__":
    main()
