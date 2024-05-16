import pandas as pd
import numpy as np
from app.utils.constants import ModelColumns
import plotly.express as px
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

classnames = ["0", "1"]


def main():
    df = pd.read_csv(f"app/data/jobs-to-balance-data.csv")

    X = df.drop(ModelColumns.target, axis=1).copy()
    y = df[ModelColumns.target].copy()

    unique, counts = np.unique(y, return_counts=True)

    fig = px.bar(X, x=unique, y=counts)
    fig.update_layout(
        title="Розподіл класів початкового набору даних",
        xaxis=dict(title="Класи", tickvals=unique, ticktext=classnames),
        yaxis_title="Кількість",
        font={"size": 18},
    )
    fig.write_image(
        f"app/utils/figures/dataset_balancing/original.png",
        scale=4,
        width=1000,
        height=500,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3
    )

    unique, counts = np.unique(y_train, return_counts=True)

    fig = px.bar(X_train, x=unique, y=counts)
    fig.update_layout(
        title="Розподіл класів тренувального набору даних",
        xaxis=dict(title="Класи", tickvals=unique, ticktext=classnames),
        yaxis_title="Кількість",
        font={"size": 18},
    )
    fig.write_image(
        f"app/utils/figures/dataset_balancing/unbalanced_train.png",
        scale=4,
        width=1000,
        height=500,
    )

    rus = RandomUnderSampler(random_state=42, replacement=True)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    unique, counts = np.unique(y_train, return_counts=True)

    fig = px.bar(X_train, x=unique, y=counts)
    fig.update_layout(
        title="Розподіл класів збалансованого тренувального набору даних",
        xaxis=dict(title="Класи", tickvals=unique, ticktext=classnames),
        yaxis_title="Кількість",
        font={"size": 18},
    )
    fig.write_image(
        f"app/utils/figures/dataset_balancing/balanced_train.png",
        scale=4,
        width=1000,
        height=500,
    )


if __name__ == "__main__":
    main()
