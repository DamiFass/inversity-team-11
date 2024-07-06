from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.metrics import confusion_matrix

# from utils.explore_data import compare_columns_day


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
    
def compare_columns_day(df1, df2, df3, df4 ,col, sample_per_day, day):
    fig, axis = plt.subplots(1, 1)
    axis.plot(df1[col].values[sample_per_day*(day-1):sample_per_day*day])
    axis.plot(df2[col].values[sample_per_day*(day-1):sample_per_day*day])
    axis.plot(df3[col].values[sample_per_day*(day-1):sample_per_day*day])
    axis.plot(df4[col].values[sample_per_day*(day-1):sample_per_day*day])
    axis.set_ylabel(col)
    # axis.legend()
    plt.show()


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

    col = df.columns.to_list()[0]
    sample_per_day = int(df.shape[0]/365) 
    day = 100
    compare_columns_day(df1, df2, df3, df4 ,col, sample_per_day, day)

    timepoints = [dfi.index.values[0] for dfi in [df1, df2, df3, df4]]
    savepath = abspath / "figures"
    savepath.mkdir(exist_ok=True)
    plot_all_columns(df, timepoints, savepath=savepath)

    #
    compute_turbine_status(df1)



    # plot_all_columns(df1)

    # x = df.drop(columns=["turbine_status"])
    # y = df["turbine_status"]

    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, random_state=0
    # )

    # regressor = LogisticRegression()
    # regressor.fit(x_train, y_train)

    # result = regressor.score(x_test, y_test)

    # print(result)


if __name__ == "__main__":
    main()
