##############################################################
#                                                            #
#                                                            #
#            Created by Ahmed Taibeche                       #
#                                                            #
#                                                            #
#                                                            #
##############################################################


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/processed_data.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
# if we try to plot the hole thing this wont make any sens because there are time gaps between each subset
subset_df = df[df["set"] == 1]
plt.plot(subset_df["acc_y"])

# plot this set using samples instead of timestamps
plt.plot(subset_df["acc_y"].reset_index(drop=True))


# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["figure.figsize"] = [15, 5]

mpl.rcParams["figure.dpi"] = 100


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df["label"].unique():
    subset_label = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset_label["acc_y"][:100].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# define a function so we can use it to plot
# --------------------------------------------------------------
def plot_fitness(data_frame, group_by, data_to_draw, x_label, y_label):
    fig, ax = plt.subplots()
    data_frame.groupby(group_by)[data_to_draw].plot(ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend()


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = (
    df.query("label =='squat'").query("participant == 'A'").reset_index(drop=True)
)

plot_fitness(category_df, "category", "acc_y", "Sample", "acc_y")

# --------------------------------------------------------------
# Compare participants to see if there's a pattern
# --------------------------------------------------------------
participants_df = (
    df.query("label == 'bench'").sort_values("participant").reset_index(drop=True)
)

plot_fitness(participants_df, "participant", "acc_y", "Sample", "acc_y")

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
tag_label = "squat"
tag_participant = "A"

all_axis_df = (
    df.query(f"label == '{tag_label}'")
    .query(f"participant == '{tag_participant}'")
    .reset_index(drop=True)
)


plot_fitness(all_axis_df, "participant", ["acc_x", "acc_y", "acc_z"], "Sample", "acc")


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
tags_label = sorted(list(df["label"].unique()))
tags_participant = sorted(list(df["participant"].unique()))

# Accelerometer data visuaulization
for label_x in tags_label:
    for participant_x in tags_participant:
        # define the data frame for each label of the participant X
        all_axis_df = (
            df.query(f"label == '{label_x}'")
            .query(f"participant == '{participant_x}'")
            .reset_index(drop=True)
        )

        if len(all_axis_df) > 0:
            plot_fitness(
                all_axis_df,
                "participant",
                ["acc_x", "acc_y", "acc_z"],
                "Sample",
                "Accelerometer",
            )
            plt.title(f"{label_x} ({participant_x})")

# gyroscope data visuaulization
for label_x in tags_label:
    for participant_x in tags_participant:
        # define the data frame for each label of the participant X
        all_axis_df = (
            df.query(f"label == '{label_x}'")
            .query(f"participant == '{participant_x}'")
            .reset_index(drop=True)
        )

        if len(all_axis_df) > 0:
            plot_fitness(
                all_axis_df,
                "participant",
                ["gyr_x", "gyr_y", "gyr_z"],
                "Sample",
                "Gyroscope",
            )
            plt.title(f"{label_x} ({participant_x})")


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
tag_label = "row"
tag_participant = "A"

combined_plot_df = (
    df.query(f"label == '{tag_label}'")
    .query(f"participant == '{tag_participant}'")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")
# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
tags_label = sorted(list(df["label"].unique()))
tags_participant = sorted(list(df["participant"].unique()))

# Accelerometer data visuaulization
for label_x in tags_label:
    for participant_x in tags_participant:
        # define the data frame for each label of the participant X
        combined_plot_df = (
            df.query(f"label == '{label_x}'")
            .query(f"participant == '{participant_x}'")
            .reset_index(drop=True)
        )

        if len(combined_plot_df) > 0:

            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("samples")
            plt.savefig(
                f"../../reports/figures/{label_x.title()} ({participant_x}).png"
            )
            plt.show()
