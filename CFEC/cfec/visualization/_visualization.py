import numpy as np
import pandas as pd

from typing import Optional, List, Any
from numpy.typing import NDArray
from cfec.constraints import ValueMonotonicity, Freeze, OneHot, ValueNominal


def show(x: pd.Series, cf: pd.Series, constraints: Optional[List[Any]] = None) -> pd.DataFrame:
    cf = cf.round(3)
    x = x.round(3)
    df = pd.concat([x, cf.transpose()], axis=1)
    df.columns = ["X", "X'"]
    df["index"] = list(range(len(x)))
    df = df[df["X"] != df["X'"]]
    df["change"] = df["X'"] - df["X"]

    if constraints is not None:
        df = _check_constraints(df, constraints)

    return df.drop(["index"], axis=1)


def compare(x: pd.Series, cfs: pd.DataFrame, constraints: Optional[List[Any]] = None,
            method_names: Optional[List[str]] = None) -> pd.DataFrame:
    cfs = cfs.round(3)
    x = x.round(3)
    constraints = [] if constraints is None else constraints
    df = pd.concat([x, cfs.T], axis=1)
    df.columns = ["X"] + ["CF" + str(i) + " change" for i in range(1, df.shape[1])]
    for col in df.iloc[:, 1:]:
        df[col] = df[col] - df["X"]

    different = _count_difference_number(df, constraints)
    df_info = pd.DataFrame([different], columns=df.columns, index=["number of attributes changed"])
    if method_names is not None and len(method_names) == len(different) - 1:
        df_info.loc["method"] = ["-"] + method_names
    df_info["constraint"] = ""
    _add_constraints_column(df, constraints)
    for col in df.iloc[:, 1:]:
        df[col].replace(0, "-", inplace=True)

    df_final = pd.concat([df_info, df], axis=0)

    return df_final


def _count_difference_number(df: pd.DataFrame, constraints: Optional[List[Any]] = None) -> NDArray[np.float64]:
    differences = np.zeros(df.shape[1])
    if constraints is None:
        constraints = []
    for i, col in enumerate(df.iloc[:, 1:]):
        diff = df[col].copy()
        for constraint in constraints:
            if isinstance(constraint, OneHot):
                feature = diff.iloc[constraint.start_column:constraint.end_column + 1]
                if sum(feature) != 0:
                    differences[i + 1] += 1
                diff.drop(feature.index, inplace=True)
        differences[i + 1] += sum(diff != 0)
    return differences


def _add_constraints_column(df: pd.DataFrame, constraints: Optional[List[Any]] = None):
    if constraints is None:
        constraints = []
    df["constraint"] = ""
    for constraint in constraints:
        if isinstance(constraint, OneHot):
            df.at[constraint.start_column:constraint.end_column + 1, "constraint"] = "OneHot"
        elif isinstance(constraint, ValueMonotonicity):
            df.at[constraint.columns, "constraint"] = "Monotonicity " + constraint.direction
        elif isinstance(constraint, ValueNominal):
            df.at[constraint.columns, "constraint"] = "ValueNominal"
        elif isinstance(constraint, Freeze):
            df.at[constraint.columns, "constraint"] = "Freeze"


def _check_constraints(df: pd.DataFrame, constraints: Optional[List[Any]] = None) -> pd.DataFrame:
    if constraints is None:
        constraints = []
    df["constraint"] = ""
    for constraint in constraints:
        if isinstance(constraint, OneHot):
            changed = df[df["index"].between(constraint.start_column, constraint.end_column)]
            # if len(changed) == 2:
            #     value_original = changed["X"][changed["X"] == 1].index.tolist()[0]
            #     value_cf = changed["X'"][changed["X'"] == 1].index.tolist()[0]
            #     df.loc[constraint.name] = [value_original, value_cf, -1, value_cf, "OneHot"]
            #     df.drop(changed.index, inplace=True)

        elif isinstance(constraint, ValueMonotonicity):
            for column in constraint.columns:
                if column in df.index:
                    df.at[column, "constraint"] = "Monotonicity " + constraint.direction
                    if (constraint.direction == "increasing" and df.loc[column]["change"] < 0) or (
                            constraint.direction == "decreasing" and df.loc[column]["change"] > 0):
                        df.at[column, "constraint"] = df.at[column, "constraint"] + "*"

        elif isinstance(constraint, ValueNominal):
            for column in constraint.columns:
                if column in df.index:
                    df.at[column, "constraint"] = "ValueNominal"

        elif isinstance(constraint, Freeze):
            for column in constraint.columns:
                if column in df.index:
                    df.at[column, "constraint"] = "Freeze*"
    return df
