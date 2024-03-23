import pandas as pd
from glob import glob


# List all data in data/raw/MetaMotion
files = glob("../../data/raw/MetaMotion/*.csv")


def process_data_from_files(files):
    # Extract features from filename
    file_path = "../../data/raw/MetaMotion\\"

    # creating two dataframes to merge all the accelerometer (gyroscope) df into one
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    # create sets to distinguish between parts of the data frames (for later use)
    acc_set = 1
    gyr_set = 1

    # looping through all files to extract features and to set the sets
    for f in files:
        participant = f.split("-")[0].replace(file_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123_MetaWear_2019")

        # read the csv file and then add the extracted features into the df
        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # merge all records of accelerometer into one dataframe
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        # merge all records of accelerometer into one dataframe
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    # converting the epoch (unix time) into datetime object and seting the index to it
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # deleting unneeded time columns
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]
    # gyroscope dataframe
    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


acc_df, gyr_df = process_data_from_files(files)
# --------------------------------------------------------------

# Merging datasets
merged_data_df = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

# rename columns
merged_data_df.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]
# --------------------------------------------------------------

# Resample data (frequency conversion)
# --------------------------------------------------------------
# lot of rows has eithre acc NONE values or gyr NONE values so we are going to resample it
# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

# if we apply the ressampling directly this will result a big issue as some days does not have records
days = [g for n, g in merged_data_df.groupby(pd.Grouper(freq="D"))]

data_ressampled_df = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_ressampled_df["set"] = data_ressampled_df["set"].astype("int")

data_ressampled_df.info()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
# we are gonna export it again into pickle file so we dont need to convert the time stamps again
data_ressampled_df.to_pickle("../../data/interim/processed_data.pkl")
