from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_all_columns(df, timepoints, savepath=None):
    # dimensions for a 16:9 slide
    width_in_inches = 13.33
    height_in_inches = 7.5
    # plotting
    n_cols = len(df.columns.values)
    fig, axes = plt.subplots(n_cols, 1, figsize=(width_in_inches, height_in_inches))
    for i, (col, axis) in enumerate(zip(df.columns.values, axes.flat)):
        axis.plot(df.index.values, df[col])
        axis.set_ylabel(col)
        if i == n_cols - 1:
            axis.set_xlabel("Date")
        for ti in timepoints:
            axis.axvline(ti, ls="--", c="r")
    fig.tight_layout()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath / "figure.png", dpi=600)
        plt.close()


def fix_time_column(df):
    time_format = "%d %m %Y %H:%M"
    df["Date/Time"] = (
        df["Date/Time"].astype(str).apply(lambda x: datetime.strptime(x, time_format))
    )
    df.set_index("Date/Time", inplace=True)


def compute_turbine_status(df, threshold=10):
    # Defining Power deficit values in percentage
    power_deficit_percentage = (
        100
        * (df["LV ActivePower (kW)"] - df["Theoretical_Power_Curve (KWh)"])
        / df["Theoretical_Power_Curve (KWh)"]
    )
    df.insert(4, "power_deficit_percentage", power_deficit_percentage)

    # considering 0 for zero values of theoretical power
    df.loc[df["Theoretical_Power_Curve (KWh)"] == 0, "power_deficit_percentage"] = 0

    # Defining the status label of the turbines wih two numeric values: 1 for "ok"; 0 for "not ok"
    condition = (df["power_deficit_percentage"] < threshold) & (
        df["power_deficit_percentage"] > -threshold
    )
    df.loc[condition, "turbine_status"] = (
        1  # meaning that the turbine don't work near theoretical power curve
    )
    condition = (df["power_deficit_percentage"] > threshold) | (
        df["power_deficit_percentage"] < -threshold
    )
    df.loc[condition, "turbine_status"] = (
        0  # meaning that the turbine does work near theoretical power curve
    )


def compare_columns_day(df1, df2, df3, df4, col="Wind Speed (m/s)"):
    fig, axis = plt.subplots(1, 1)
    axis.plot(df1[col].values[138 * (20 - 1) : 138 * 20])
    axis.plot(df2[col].values[138 * (20 - 1) : 138 * 20])
    axis.plot(df3[col].values[138 * (20 - 1) : 138 * 20])
    axis.plot(df4[col].values[138 * (20 - 1) : 138 * 20])
    axis.set_ylabel(col)
    plt.show()


def train_model(df):
    x = df.drop(columns=["turbine_status"])
    y = df["turbine_status"]
    classifier = RandomForestClassifier()
    classifier.fit(x, y)
    return classifier


def main():
    abspath = Path(os.getcwd())

    loadpath = abspath / "data"
    df = pd.read_csv(loadpath / "T1.csv")

    fix_time_column(df)

    # timeframe min-max
    start_time = df.index.min()
    end_time = df.index.max()
    # generate time intervals
    time_ranges = pd.date_range(start=start_time, end=end_time, periods=5)
    # slice the DataFrame into 4 parts based on these intervals
    df1 = df[(df.index >= time_ranges[0]) & (df.index < time_ranges[1])]
    df2 = df[(df.index >= time_ranges[1]) & (df.index < time_ranges[2])]
    df3 = df[(df.index >= time_ranges[2]) & (df.index < time_ranges[3])]
    df4 = df[(df.index >= time_ranges[3]) & (df.index <= time_ranges[4])]

    df_list = [df1, df2, df3, df4]
    compare_columns_day(df1, df2, df3, df4)

    timepoints = [dfi.index.values[0] for dfi in df_list]
    savepath = abspath / "figures"
    savepath.mkdir(exist_ok=True)
    plot_all_columns(df, timepoints, savepath=savepath)

    for df in df_list:
        compute_turbine_status(df)

    model = train_model(df_list[0])

    for df in df_list[1:]:
        x_test = df.drop(columns=["turbine_status"])
        y_test = df["turbine_status"]
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()


if __name__ == "__main__":
    main()