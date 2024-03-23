import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/outliers_removed.pkl")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()
# --------------------------------------------------------------------
# Calculating set duration and the average time for each repetition
# for each category because medium workout has 10 reps and 5 for heavy
# --------------------------------------------------------------------
df[df.set == 50]["acc_y"].plot()

for s in df["set"].unique():
    # calculate the duration
    start = df[df.set == s].index[0]
    stop = df[df.set == s].index[-1]
    duration = stop - start

    # add a new column with the duration of each set
    df.loc[df.set == s, "duration"] = duration.seconds

# create a data frame with the average of duration of each category
duration_df = df.groupby(["category"])["duration"].mean()

avg_time_rep_Hv = duration_df.iloc[0] / 5
avg_time_rep_Md = duration_df.iloc[1] / 10


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
# this is an instance of the class LowPassfilter so we can use the low pass function
LowPass = LowPassFilter()
fs = 1000 / 200
cutoff = 1.2

# experiment on one color
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)
subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

# plot to see the difference and try to find out the best value for cutoff
# so we don't lose the pattern and apply the filter in the same time
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# after setting cutoff freq to 1.2 we apply this to all of the columns
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
