import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

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
df_temporal = pd.concat(df_temporal_list)

# see the results
subset[["acc_y", "acc_y_mean_ws_5", "acc_y_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_mean_ws_5", "gyr_y_std_ws_5"]].plot()
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
# we need to drop the time index to use the frequency terminology
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

# define the sampling rate which expresses the number of samples per second
fs = int(1000 / 200)
ws = int(2800 / 200)

# test for one set of the data
df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
# Visualize results
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

df_freq_list = []

for s in df_freq["set"].unique():
    print(f"Applying Fourier transformations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)


df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
df_freq.info()
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# the data is highly correlated because we used ws of 14 thus could cause overfitting
# so we are going to deal with that we will drop Nan values and
# it's recommanded to drop 50% if we have enough data
df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2]
for x in df_freq.columns.unique():
    print(x)
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = list(df_freq.columns.unique()[:3])
k_values = range(2, 10)
inertias = []
for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

# plot to use the elbow technique to get the right number of clusters
plt.figure(figsize=(20, 10))
plt.plot(k_values, inertias)
plt.xlabel("K")
plt.ylabel("sum of the squared distances")
plt.show()

# from the graph we can see that k=5
subset = df_cluster[cluster_columns]
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot clusters grouping clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# plot accelemoeter data based on label to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/data_features.pkl")