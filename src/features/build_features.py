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
df_pca = df_lowpass.copy()
# create an instance PCA of PrincipalComponentClass
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# plot the PC to use the elbow technique in order to find the number of PCA
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variance")
plt.show()

# from the graph we can clearly see the elbow on component 3 so we are taking n_pc =3
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)


subset = df_pca[df_pca["set"] == 45]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes (calculate Scalar Magnitude)
# r= sqrt((x^2) + (y^2) + (z^2))
# --------------------------------------------------------------
df_squared = df_pca.copy()

# calcaulating acc magnitude
acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
df_squared["acc_r"] = np.sqrt(acc_r)

# calcaulating gyr magnitude
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2
df_squared["gyr_r"] = np.sqrt(gyr_r)

# plot acc_r and gyr_r
subset = df_squared[df_squared["set"] == 5]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

# it's not exact science how to set ws
# so we have to set it with trial and error
ws = int(1000 / 200)

df_temporal_list = []

for s in df_temporal["set"].unique():
    """Separates data into individual sets.

    The rolling window calculation used later depends on previous samples within
    a set. Combining data from multiple exercises or sets would lead to inaccurate
    features and unreliable results. This function ensures each set is processed
    independently.
    """

    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

# regroup all the data together back if you run df_temporal.info() you will find
# that some of the data is missing This is ok rather than mix the data together
# df_temporal = pd.concat(df_temporal_list)

# see the results
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()
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
