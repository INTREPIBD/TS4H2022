import numpy as np
import typing as t
import pandas as pd
import sklearn.utils

from timebase.data.static import *


def quality_control(
    args,
    channel_data: t.Dict[str, np.ndarray],
    sampling_rates: t.Dict[str, int],
):
    """
    Apply Kleckner et al. 2018 quality control to EDA and TEMP plus new rule on HR
    see Figure 1 in https://pubmed.ncbi.nlm.nih.gov/28976309/
    Returns:
        features: np.ndarray, the filtered recordings where 0 index is one continuous recording
    """
    eda_slope = np.gradient(channel_data["EDA"])

    # Rule 1: remove EDA not within 0.05 - 60 muS
    rule1 = (channel_data["EDA"] < 0.05) + (channel_data["EDA"] > 60)
    if args.verbose == 2:
        print(
            f"Kleckner al. 2018 quality control:\t"
            f"{rule1.sum()/len(rule1):.02f}% measurements removed "
            f"by Rule 1"
        )

    # Rule 2: remove EDA slope not within -10 - 10 muS/sec
    rule2 = (eda_slope < -10) + (eda_slope > 10)
    if args.verbose == 2:
        print(f"{rule2.sum()/len(rule2):.02f}% measurements removed by Rule 2")

    # Rule 3: remove TEMP not within 30 - 40 °C
    rule3 = (channel_data["TEMP"] < 30) + (channel_data["TEMP"] > 40)
    if args.verbose == 2:
        print(f"{rule3.sum()/len(rule3):.02f}% measurements removed by Rule 3")

    # Rule 4: EDA surrounded (within 5 sec) by invalid data according to Rule 1-3
    assert (
        len(channel_data["EDA"]) > 5 * sampling_rates["EDA"]
    ), "recording is shorter than 5 seconds, cannot apply Rule 4."
    rule4 = np.correlate(
        rule1 + rule2 + rule3,
        np.ones(shape=(5 * sampling_rates["EDA"]), dtype=np.int8),
        mode="same",
    ).astype(bool)
    if args.verbose == 2:
        print(
            f"{(rule4.sum() - (rule1.sum() + rule2.sum() + rule3.sum()))/len(rule4):.02f}% "
            f"measurements removed by Rule 4\t"
        )

    # Rule 5: remove HR that are not within 25 - 250 bpm
    # Note: this is not from Kleckner et al. 2018
    rule5 = (channel_data["HR"] < 25) + (channel_data["HR"] > 250)
    if args.verbose == 2:
        print(f"{rule5.sum()/len(rule5):.02f}% measurements removed by rule 5")

    # HR is in 1Hz, EDA and TEMP are in 4Hz
    # We need to downsampling Rule 1-4 masks to 1Hz and join with Rule 5
    total_mask = rule1 + rule2 + rule3 + rule4
    total_mask = np.reshape(total_mask, newshape=(-1, sampling_rates["EDA"]))
    # set False to rows with False and total_mask is now 1Hz
    total_mask = np.min(total_mask, axis=-1)
    total_mask = total_mask + rule5
    for channel in channel_data.keys():
        # convert 1Hz mask to channel sampling rates
        mask = np.repeat(total_mask, repeats=sampling_rates[channel], axis=0)
        channel_data[channel][mask] = np.nan

    if args.verbose == 2:
        print(
            f"{total_mask.sum()/len(total_mask):.02f}% of recordings removed upon "
            f"Quality Control"
        )


def set_unique_rec_id(args, y):
    """
    if for a given NHC at a given time (T) more than one session_code was recorded,
    assign a unique session_code throughout
    """
    for id in np.unique(y[:, Y_COLS.index("NHC")]):
        for t in np.unique(
            y[:, Y_COLS.index("time")][np.where(y[:, Y_COLS.index("NHC")] == id)[0]]
        ):
            unique_recordings = np.unique(
                y[:, Y_COLS.index("Session_Code")][
                    np.where(
                        np.where(y[:, Y_COLS.index("NHC")] == id, 1, 0)
                        + np.where(y[:, Y_COLS.index("time")] == t, 1, 0)
                        == 2
                    )[0]
                ]
            )
            if len(unique_recordings) > 1:
                y[:, Y_COLS.index("Session_Code")][
                    np.where(
                        np.where(y[:, Y_COLS.index("NHC")] == id, 1, 0)
                        + np.where(y[:, Y_COLS.index("time")] == t, 1, 0)
                        == 2
                    )[0]
                ] = unique_recordings[0]


def downsample_segments(args, x: t.Dict[str, np.ndarray], y: np.ndarray, bins: int = 5):
    """
    As the dataset is imbalanced towards segments from individuals where items
    across HDRS and YMRS are zero, we bin the vector of scales' total sum into
    10 and down-sample segments from bins weighting more than 10% of the total
    dataset
    """

    ymrs_sum_binned = pd.cut(
        y[:, Y_COLS.index("YMRS_SUM")],
        bins=bins,
        labels=np.arange(bins),
    )
    hdrs_sum_binned = pd.cut(
        y[:, Y_COLS.index("HDRS_SUM")],
        bins=bins,
        labels=np.arange(bins),
    )
    labels = np.array(
        [
            f"young{str(young)}_ham{str(ham)}"
            for young, ham in zip(ymrs_sum_binned, hdrs_sum_binned)
        ]
    )

    bin_labels, bin_counts = np.unique(labels, return_counts=True)
    bin_weights = [c / bin_counts.sum() for c in bin_counts]

    if args.downsample_mode == 0:
        return x, y
    elif args.downsample_mode == 1:
        indexes2drop = []
        heavy_bins = bin_labels[np.array(bin_weights) > 1 / len(bin_weights)]
        donwsample2bin_idx = np.argsort(bin_weights)[-(len(heavy_bins) + 1)]
        no2keep = bin_counts[donwsample2bin_idx]

        for b in heavy_bins:
            indexes_i = np.where(labels == b)[0]
            states, counts_s = np.unique(
                y[:, Y_COLS.index("status")][indexes_i], return_counts=True
            )
            state_weights = [c / counts_s.sum() for c in counts_s]
            for s, n in zip(states, np.around(no2keep * np.array(state_weights))):
                indexes_b_s = np.where(
                    np.where(labels == b, 1, 0)
                    + np.where(y[:, Y_COLS.index("status")] == s, 1, 0)
                    == 2
                )[0]
                ids, counts_ids = np.unique(
                    y[:, Y_COLS.index("Session_Code")][indexes_b_s],
                    return_counts=True,
                )
                ids_weights = [c / counts_ids.sum() for c in counts_ids]
                for i, i_w in zip(ids, ids_weights):
                    indexes_b_s_i = np.where(
                        np.where(labels == b, 1, 0)
                        + np.where(y[:, Y_COLS.index("status")] == s, 1, 0)
                        + np.where(y[:, Y_COLS.index("Session_Code")] == i, 1, 0)
                        == 3
                    )[0]

                    idx2drop = list(
                        sklearn.utils.shuffle(indexes_b_s_i, random_state=123)[
                            int(n * i_w) :
                        ]
                    )
                    indexes2drop.extend(idx2drop)
        indexes2keep = list(
            set(list(np.arange(len(labels)))).difference(set(indexes2drop))
        )
        indexes2keep.sort()
        if args.verbose:
            print(
                f"{100 * len(indexes2drop)/len(y):.02f}% of segments dropped "
                f"upon downsampling"
            )
        return [x[i] for i in indexes2keep], y[indexes2keep]
    else:
        raise NotImplementedError(
            f"split_mode {args.split_mode} has not been implemented."
        )
