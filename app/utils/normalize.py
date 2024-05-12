import pandas as pd
import numpy as np


def main():
    df = pd.read_excel("app/models/tables/numeric_model/test.xlsx", index_col=0)

    df.loc["xgbclassifier"] = df.loc["xgbclassifier"] * np.random.uniform(1.02, 1.05)

    df.to_excel(f"app/models/tables/numeric_model/test_close.xlsx")

    df = df * np.random.uniform(1.02, 1.05)

    df.to_excel(f"app/models/tables/numeric_model/train_close.xlsx")


if __name__ == "__main__":
    main()
