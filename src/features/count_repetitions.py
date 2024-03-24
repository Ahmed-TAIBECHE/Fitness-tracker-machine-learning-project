##############################################################
#                                                            #
#                                                            #
#            Created by Ahmed Taibeche                       #
#                                                            #
#                                                            #
#                                                            #
##############################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# we are going back to data_processed to see the features that
# are relevant to counting the repetitions we can not use the
# data where we extracted a lot of features because we threw
# a lot of data due to overlaping and done a lot of manipulation
# of it so it's a good decision to go back
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/processed_data.pkl")
# throw irrelevant data for this part
# (we dont need to count repetition for rest DUHHH!)
df = df[df["label"] != "rest"]

# adding back the Scalar Magnitude can be useful
# calcaulating acc magnitude
acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)

# calcaulating gyr magnitude
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df = df[df["label"] == "bench"]
dead_df = df[df["label"] == "dead"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
squat_df = df[df["label"] == "squat"]


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
plot_df = squat_df
plot_df[plot_df["set"] == plot_df.set[1]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df.set[1]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df.set[1]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df.set[1]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df.set[1]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df.set[1]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df.set[1]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df.set[1]]["gyr_r"].plot()


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000 / 200
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set = bench_df[bench_df["set"] == bench_df.set[0]]
squat_set = squat_df[squat_df["set"] == squat_df.set[1]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df.set[1]]
row_set = row_df[row_df["set"] == row_df.set[1]]
dead_set = dead_df[dead_df["set"] == dead_df.set[1]]

column = "acc_r"
LowPass.low_pass_filter(
    bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.45, order=10
)
bench_set["acc_r_lowpass"].plot()


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
def count_reps(data, column="acc_r", fs=5, cutoff=0.4, order=10):
    dataset = LowPass.low_pass_filter(
        data, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    indexes = argrelextrema(dataset[column + "_lowpass"].values, np.less)
    minimum = dataset.iloc[indexes]

    # plot the results
    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(minimum[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f" {column} lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(minimum)} Reps")
    plt.show()

    return len(minimum)


reps = count_reps(bench_set)
# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
