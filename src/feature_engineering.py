import pandas as pd
from itertools import combinations


def add_features(X):
    new_X = X.copy()

    # Store new features in a dict first
    new_cols = {}

    for col1, col2 in combinations(new_X.columns, 2):
        new_col_name = f"{col1}*{col2}"
        new_cols[new_col_name] = X[col1] * X[col2]

    new_X = pd.concat([new_X, pd.DataFrame(new_cols)], axis=1)

    return new_X
