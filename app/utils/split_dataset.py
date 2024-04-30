import pandas as pd
from app.utils.constants import data_path, ModelColumns
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle


def main():
    df = pd.read_csv(f"{data_path}/jobs-main-data.csv")

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
    y = df[ModelColumns.target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3
    )

    rus = RandomUnderSampler(random_state=42, replacement=True)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    X_train.to_csv(f"{data_path}/X_train.csv", index=False)
    X_test.to_csv(f"{data_path}/X_test.csv", index=False)
    y_train.to_csv(f"{data_path}/y_train.csv", index=False)
    y_test.to_csv(f"{data_path}/y_test.csv", index=False)


if __name__ == "__main__":
    main()
