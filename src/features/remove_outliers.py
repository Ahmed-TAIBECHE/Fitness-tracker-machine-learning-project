import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from plot_binary_outliers import *
from outlier_detection_functions import *

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/processed_data.pkl")

columns_outliers = list(df.columns.unique()[:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["lines.linewidth"] = 5
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["figure.dpi"] = 100

df[columns_outliers[:3] + ["label"]].boxplot(by="label", layout=(1, 3))
df[columns_outliers[3:] + ["label"]].boxplot(by="label", layout=(1, 3))


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Loop over all columns
for col in columns_outliers:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution
df[columns_outliers[:3] + ["label"]].plot.hist(
    by="label", figsize=(20, 10), layout=(3, 3)
)
df[columns_outliers[3:] + ["label"]].plot.hist(
    by="label", figsize=(20, 10), layout=(3, 3)
)

"""after analyzing the results we can observe that for the most of the data
    its normally distributed so we are gonna apply this on our data
"""

# Loop over all columns
for col in columns_outliers:
    dataset = mark_outliers_chauvenet(dataset, col, C=2)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------
"""I am going to choose the chauvenets criterion because it does not
    consider a lot of data as outliers in contrary of IQR METHOD"""
# Test on single column
col = "gyr_z"
mark_outliers_chauvenet(dataset, col)
dataset[dataset["gyr_z_outlier"]]

# replacing the outliers with NaN
dataset.loc[dataset["gyr_z_outlier"], "gyr_z"] = np.nan

# Create a loop
outliers_removed_df = df.copy()

for col in columns_outliers:
    for label in df["label"].unique():
        # mark the outliers per exercice rather the whole data for better results
        dataset = mark_outliers_chauvenet(df[df["label"] == label], col)

        # replacing the outliers with NaN
        dataset.loc[dataset[col + "_outlier"], col] = np.nan

        # update the column in the original dataframe
        outliers_removed_df.loc[(outliers_removed_df["label"] == label), col] = dataset[
            col
        ]

        # print the number of outliners removed from each column for each label
        n_outliners = len(dataset[col]) - len(dataset[col].dropna())
        print(f"{n_outliners} removed from {col} in {label}")


outliers_removed_df.info()
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outliers_removed_df.to_pickle("../../data/interim/outliers_removed.pkl")
