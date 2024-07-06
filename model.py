from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


def main():
    loadpath = Path(os.getcwd()) / "data"

    df = pd.read_csv(loadpath / "T1.csv")
    print(df.columns.values)

    # Define the time format
    time_format = "%d %m %Y %H:%M"

    # Convert the "Date/Time" column to string and then apply datetime conversion
    df["Date/Time"] = (
        df["Date/Time"].astype(str).apply(lambda x: datetime.strptime(x, time_format))
    )
    # Print the modified DataFrame
    df.set_index("Date/Time", inplace=True)

    fig, axes = plt.subplots(len(df.columns.values), 1)
    for col, axis in zip(df.columns.values, axes.flat):
        axis.plot(df.index.values, df[col])
        axis.set_ylabel(col)
    fig.tight_layout()
    plt.show()

    # Defining Power deficit values in percentage
    power_deficit_percentage = (
        100
        * (df["LV ActivePower (kW)"] - df["Theoretical_Power_Curve (KWh)"])
        / df["Theoretical_Power_Curve (KWh)"]
    )
    df.insert(4, "power_deficit_percentage", power_deficit_percentage)

    # considering 0 for zero values of theoritcal power
    df.loc[df["Theoretical_Power_Curve (KWh)"] == 0, "power_deficit_percentage"] = 0

    # Defining the status label of the turbines wih two numeric values: 1 for "ok"; 0 for "not ok"
    threshold = 10

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

    print(df.head())

    # fig, axis = plt.subplots(1, 1)
    # axis.plot(df.index.values, df["turbine_status"])
    # plt.show()

    x = df.drop(columns=["turbine_status"])
    y = df["turbine_status"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    regressor = LogisticRegression()
    regressor.fit(x_train, y_train)

    result = regressor.score(x_test, y_test)

    print(result)


if __name__ == "__main__":
    main()
