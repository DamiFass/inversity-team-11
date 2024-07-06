from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# data_folder = '../data/'
# for filename in os.listdir(data_folder):
#     file_path = os.path.join(data_folder, filename)
#     if os.path.isfile(file_path):
#         print(file_path)
        
df = pd.read_csv('../data/T2.csv')

print(df.columns.values)

# Define the time format
time_format = "%d %m %Y %H:%M"

# Convert the "Date/Time" column to string and then apply datetime conversion
df["Date/Time"] = (
    df["Date/Time"].astype(str).apply(lambda x: datetime.strptime(x, time_format))
)
# Print the modified DataFrame
df.set_index("Date/Time", inplace=True)

column_names = df.columns.to_list()

sample_per_day = int(df.shape[0]/365)
col = column_names[1]
day = 100


def plot_column_day(col, sample_per_day, day):
    fig, axis = plt.subplots(1, 1)
    axis.plot(df[col].values[sample_per_day*(day-1):sample_per_day*day])
    axis.set_ylabel(col)
    plt.show()
  
T1 = pd.read_csv('../data/T1.csv')
T2 = pd.read_csv('../data/T2.csv')
T3 = pd.read_csv('../data/T3.csv')
T4 = pd.read_csv('../data/T4.csv')
T5 = pd.read_csv('../data/T5.csv')

def compare_columns_day(df1, df2, df3, df4, df5 ,col, sample_per_day, day):
    fig, axis = plt.subplots(1, 1)
    axis.plot(df1[col].values[sample_per_day*(day-1):sample_per_day*day], c='k', label='Real')
    axis.plot(df2[col].values[sample_per_day*(day-1):sample_per_day*day], alpha=0.5)
    axis.plot(df3[col].values[sample_per_day*(day-1):sample_per_day*day], alpha=0.5)
    axis.plot(df4[col].values[sample_per_day*(day-1):sample_per_day*day], alpha=0.5)
    axis.plot(df5[col].values[sample_per_day*(day-1):sample_per_day*day], alpha=0.5)
    axis.set_ylabel(col)
    axis.legend()
    plt.show()
    
compare_columns_day(T1, T2, T3, T4, T5, col, sample_per_day, day)  
    
plot_column_day(col, sample_per_day, day)
    