"""
Helper functions to preprocess CSV files
Reference on data export and formatting of Empatica E4 wristband
https://support.empatica.com/hc/en-us/articles/201608896-Data-export-and-formatting-from-E4-connect-
"""

import mne
import warnings
import typing as t
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from copy import deepcopy
from math import ceil, floor

from timebase.data.static import *
from timebase.data import utils, filter_data
from timebase.utils.utils import update_dict
from sklearn.model_selection import train_test_split

warnings.simplefilter("error", RuntimeWarning)


def select_items(args):
    """select which item(s) should be set as target(s)"""
    assert (0 not in args.ymrs) or (
        0 not in args.ymrs
    ), "At least one item should be selected as target"
    if len(set(args.ymrs).difference(set(range(0, 12)))):
        raise Exception(f"{args.ymrs} not in [0,11]")
    if len(set(args.hdrs).difference(set(range(0, 18)))):
        raise Exception(f"{args.ymrs} not in [0,17]")
    if 0 in args.ymrs:
        ymrs_indexes = []
        ymrs_selected = []
    else:
        ymrs_indexes = [i - 1 for i in args.ymrs]
        ymrs_selected = [f"YMRS{i}" for i in args.ymrs]
    if 0 in args.hdrs:
        hdrs_indexes = []
        hdrs_selected = []
    else:
        hdrs_indexes = [i + len(YMRS_item_ranks) - 1 for i in args.hdrs]
        hdrs_selected = [f"HDRS{i}" for i in args.hdrs]
    indexes = ymrs_indexes + hdrs_indexes
    args.selected_items = ymrs_selected + hdrs_selected
    args.rank_normalizer = tf.convert_to_tensor(
        np.asarray(RANK_NORMALIZER)[indexes], dtype=tf.float32
    )
    item_ranks = np.asarray(YMRS_item_ranks + HDRS_item_ranks)
    args.item_ranks = tf.convert_to_tensor(item_ranks[indexes], dtype=tf.float32)
    item_max = np.asarray(MAX_YMRS + MAX_HDRS)
    args.item_max = tf.convert_to_tensor(item_max[indexes], dtype=tf.float32)


def read_clinical_info(filename: str):
    """Read clinical EXCEL file"""
    assert os.path.isfile(filename), f"clinical file {filename} does not exists."
    xls = pd.ExcelFile(filename)
    info = pd.read_excel(xls, sheet_name=None)  # read all sheets
    return pd.concat(info)


def low_pass_filter(recording: np.ndarray, sampling_rate: int):
    return mne.filter.filter_data(
        data=recording.astype(np.float64),
        sfreq=sampling_rate,
        l_freq=0,
        h_freq=0.35,
        filter_length=257,
        verbose=False,
    ).astype(np.float32)


def split_acceleration(
    channel_data: t.Dict[str, np.ndarray],
    sampling_rates: t.Dict[str, int],
):
    """Split 3D ACC into ACC_x, ACC_y and ACC_z"""
    channel_data["ACC_x"] = channel_data["ACC"][:, 0]
    channel_data["ACC_y"] = channel_data["ACC"][:, 1]
    channel_data["ACC_z"] = channel_data["ACC"][:, 2]
    del channel_data["ACC"]
    sampling_rates["ACC_x"] = sampling_rates["ACC"]
    sampling_rates["ACC_y"] = sampling_rates["ACC"]
    sampling_rates["ACC_z"] = sampling_rates["ACC"]
    del sampling_rates["ACC"]


def segmentation(
    args,
    session_data: t.Dict[str, np.ndarray],
    channel_freq: t.Dict[str, int],
) -> (t.Dict[str, np.ndarray], int):
    """
    Segment preprocessed features along the temporal dimension into
    N non-overlapping segments where each segment has size args.segment_length
    Return:
        data: t.Dict[str, np.ndarray]
                dictionary of np.ndarray, where the keys are the channels
                and each np.ndarray are in shape (num. segment, window size)
        size: int, number of extracted segments
    """
    assert (segment_length := args.segment_length) > 0
    channels = list(session_data.keys())
    channel_segments = {c: [] for c in channels}

    # segment each channel using a sliding window that space out equally
    for channel in channels:
        window_size = segment_length * channel_freq[channel]
        recording = session_data[channel]
        num_segments = floor(len(recording) / window_size)
        indexes = np.linspace(
            start=0,
            stop=len(recording) - window_size,
            num=num_segments,
            dtype=int,
        )
        channel_segments[channel].extend(
            [recording[i : i + window_size, ...] for i in indexes]
        )

    num_channel_segments = [len(s) for s in channel_segments.values()]
    assert (
        len(set(num_channel_segments)) == 1
    ), "all channels must have equal length after segmentation"
    # dictionary of list of np.ndarray, where channel are the keys.
    data = {c: [] for c in channels}
    for i in range(num_channel_segments[0]):
        segment, drop = {}, False
        for channel in channels:
            recording = channel_segments[channel][i]
            # drop segment with NaN values
            if np.isnan(recording).any():
                drop = True
                break
            segment[channel] = recording
        if not drop:
            for channel in channels:
                data[channel].append(segment[channel])
    # ensure each channel has equal number of segments
    sizes = [len(data[c]) for c in channels]
    assert len(set(sizes)) == 1, "unequal number of extracted segments."
    data = {c: np.asarray(r) for c, r in data.items()}
    return data, sizes[0]


def load_channel(recording_dir: str, channel: str):
    """Load channel CSV data from file
    Returns
      unix_t0: int, the start time of the recording in UNIX time
      sampling_rate: int, sampling rate of the recording (if exists)
      data: np.ndarray, the raw recording data
    """
    assert channel in CSV_CHANNELS, f"unknown channel {channel}"
    raw_data = pd.read_csv(
        os.path.join(recording_dir, f"{channel}.csv"), delimiter=",", header=None
    ).values

    unix_t0, sampling_rate, data = None, -1.0, None
    if channel == "IBI":
        unix_t0 = raw_data[0, 0]
        data = raw_data[1:]
    else:
        unix_t0 = raw_data[0] if raw_data.ndim == 1 else raw_data[0, 0]
        sampling_rate = raw_data[1] if raw_data.ndim == 1 else raw_data[1, 0]
        data = raw_data[2:]
    assert sampling_rate.is_integer(), "sampling rate must be an integer"
    data = np.squeeze(data)
    return int(unix_t0), int(sampling_rate), data.astype(np.float32)


def pad(args, data: np.ndarray, sampling_rate: int):
    """
    Upsample channel whose sampling rate is lower than args.time_alignment
    """

    # trim additional recordings that does not make up a second.
    data = data[: data.shape[0] - (data.shape[0] % sampling_rate)]

    s_shape = [data.shape[0] // sampling_rate, sampling_rate]
    p_shape = [s_shape[0], args.time_alignment]  # padded shape
    o_shape = [s_shape[0] * args.time_alignment]  # output shape
    if len(data.shape) > 1:
        s_shape.extend(data.shape[1:])
        p_shape.extend(data.shape[1:])
        o_shape.extend(data.shape[1:])
    # reshape data s.t. the 1st dimension corresponds to one second
    s_data = np.reshape(data, newshape=s_shape)

    # calculate the padding value
    if args.padding_mode == "zero":
        pad_value = 0
    elif args.padding_mode == "last":
        pad_value = s_data[:, -1, ...]
        pad_value = np.expand_dims(pad_value, axis=1)
    elif args.padding_mode == "average":
        pad_value = np.mean(s_data, axis=1, keepdims=True)
    elif args.padding_mode == "median":
        pad_value = np.median(s_data, axis=1, keepdims=True)
    else:
        raise NotImplementedError(
            f"padding_mode {args.padding_mode} has not been implemented."
        )

    padded_data = np.full(shape=p_shape, fill_value=pad_value, dtype=np.float32)
    padded_data[:, :sampling_rate, ...] = s_data
    padded_data = np.reshape(padded_data, newshape=o_shape)
    return padded_data


def pool(args, data: np.ndarray, sampling_rate: int):
    """
    Downsample channel whose sampling rate is greater than args.time_alignment
    """
    size = data.shape[0] - (data.shape[0] % sampling_rate)
    shape = (
        size // int(sampling_rate / args.time_alignment),
        int(sampling_rate / args.time_alignment),
    )
    if data.ndim > 1:
        shape += (data.shape[-1],)
    data = data[:size]
    new_data = np.reshape(data, newshape=shape)
    # apply pooling on the axis=1
    if args.downsampling == "average":
        new_data = np.mean(new_data, axis=1)
    elif args.downsampling == "max":
        new_data = np.max(new_data, axis=1)
    else:
        raise NotImplementedError(f"unknown downsampling method {args.downsampling}.")
    return new_data


def trim(data: np.ndarray, sampling_rate: int):
    """
    Trim, if necessary, tail of channel whose sampling rate is equal to
    args.time_alignment
    """
    size = data.shape[0] - (data.shape[0] % sampling_rate)
    return data[:size]


def resample(args, data: np.ndarray, sampling_rate: int):
    """
    Resample data so that channels are time aligned based on the required no of
    cycles per second (args.time_alignment)
    """
    ratio = args.time_alignment / sampling_rate
    if ratio > 1:
        new_data = pad(args, data, sampling_rate)
    elif ratio < 1:
        new_data = pool(args, data, sampling_rate)
    else:
        new_data = trim(data, sampling_rate)
    return new_data


def resample_channels(
    args,
    channel_data: t.Dict[str, np.ndarray],
    sampling_rates: t.Dict[str, int],
):
    data_freq = deepcopy(sampling_rates)
    if args.time_alignment:
        for channel, recording in channel_data.items():
            channel_data[channel] = resample(
                args, data=recording, sampling_rate=sampling_rates[channel]
            )
            data_freq[channel] = args.time_alignment
    return data_freq


def preprocess_channel(recording_dir: str, channel: str):
    """
    Load and downsample channel using args.downsampling s.t. each time-step
    corresponds to one second in wall-time
    """
    assert channel in CSV_CHANNELS and channel != "IBI"
    _, sampling_rate, data = load_channel(recording_dir=recording_dir, channel=channel)
    # transform to g for acceleration
    if channel == "ACC":
        data = data * 2 / 128
    # despike, apply filter on EDA and TEMP data
    # note: kleckner2018 uses a length of 2057 for a signal sampled at 32Hz,
    # EDA from Empatica E4 is sampled at 4Hz (1/8)
    if channel == "EDA" or channel == "TEMP":
        data = low_pass_filter(recording=data, sampling_rate=sampling_rate)
    if channel != "HR":
        # HR begins at t0 + 10s, remove first 10s from channels other than HR
        data = data[sampling_rate * HR_OFFSET :]
    return data, sampling_rate


def remove_zeros(recordings: np.ndarray, threshold: int = 5) -> t.List[np.ndarray]:
    """
    Remove recordings where all channels contain 0s for longer than threshold
    time-steps
    Args:
      recordings: np.ndarray
      threshold: int, the threshold (in time-steps) where channels can contain 0s
    Return:
      features: np.ndarray, filtered recordings where 0 index one continuous recording
    """
    assert 0 < threshold < recordings.shape[0]
    sums = np.sum(np.abs(recordings), axis=-1)
    features = []
    start, end = 0, 0
    while end < sums.shape[0]:
        if sums[end] == 0:
            current = end
            while end < sums.shape[0] and sums[end] == 0:
                end += 1
            if end - current >= threshold:
                features.append(recordings[start:current, ...])
                start = end
        end += 1
    if start + 1 < end:
        features.append(recordings[start:end, ...])
    return features


def preprocess_dir(args, recording_dir: str, clinical_info: pd.DataFrame):
    """
    Preprocess channels in recording_dir and return the preprocessed features
    and corresponding label obtained from spreadsheet.
    Returns:
      features: np.ndarray, preprocessed channels in SAVE_CHANNELS format
      label: List[int], label from clinical spreadsheet as per spreadsheet columns
    """
    session_id = int(os.path.basename(recording_dir))
    durations, channel_data, sampling_rates = [], {}, {}
    # load and preprocess all channels except IBI
    for channel in CSV_CHANNELS:
        if channel != "IBI":
            channel_data[channel], sampling_rate = preprocess_channel(
                recording_dir=recording_dir, channel=channel
            )
            durations.append(len(channel_data[channel]) // sampling_rate)
            sampling_rates[channel] = sampling_rate
    # all channels should have the same durations, but as a failsafe, crop
    # each channel to the shortest duration
    min_duration = min(durations)
    for channel, recording in channel_data.items():
        channel_data[channel] = recording[: min_duration * sampling_rates[channel]]

    split_acceleration(channel_data=channel_data, sampling_rates=sampling_rates)

    # quality control recording
    if args.qc_mode == 0:
        pass
    elif args.qc_mode == 1:
        filter_data.quality_control(
            args, channel_data=channel_data, sampling_rates=sampling_rates
        )
    else:
        raise NotImplementedError(f"QC mode {args.qc_mode} has not been implemented.")

    channel_freq = resample_channels(
        args, channel_data=channel_data, sampling_rates=sampling_rates
    )

    session_info = {
        "channel_names": utils.get_channel_names(channel_data),
        "sampling_rates": sampling_rates,
        "channel_freq": channel_freq,
    }

    try:
        session_info.update(
            {
                "min": {k: np.nanmin(v, axis=0) for k, v in channel_data.items()},
                "max": {k: np.nanmax(v, axis=0) for k, v in channel_data.items()},
                "mean": {k: np.nanmean(v, axis=0) for k, v in channel_data.items()},
                "std": {k: np.nanstd(v, axis=0) for k, v in channel_data.items()},
            }
        )
    except RuntimeWarning as e:
        print(f"Session {session_id} warning: {e}")

    # get label information
    clinical_session = clinical_info[clinical_info.Session_Code == session_id]
    if clinical_session.empty:
        raise IndexError(f"Cannot find session {session_id} in Spreadsheet.")
    label = clinical_session.values[0].tolist()

    return channel_data, label, session_info


def split_into_sets(args, x: t.Dict[str, np.ndarray], y: np.ndarray):

    if args.split_mode == 0:
        # random splits with no stratification
        idx = np.arange((len(y)))
        train_idx, test_idx = train_test_split(idx, train_size=0.7, random_state=123)
        val_idx, test_idx = train_test_split(test_idx, train_size=0.5, random_state=123)
    elif args.split_mode == 1:
        # split each session into 70:15:15 along the temporal dimension to
        # minimize overlap between segments in train/val/test
        train_idx, val_idx, test_idx = [], [], []
        session_ids, num_segments = np.unique(
            y[:, Y_COLS.index("Session_Code")], return_counts=True
        )
        for session_id in session_ids:
            indexes = np.where(y[:, Y_COLS.index("Session_Code")] == session_id)[0]
            size = len(indexes)
            if size <= 3:
                if args.verbose == 2:
                    print(
                        f"Session {int(session_id)} dropped since it has fewer "
                        f"than 4 segments"
                    )
            else:
                session_idx_train, session_idx_val, session_idx_test = np.split(
                    indexes,
                    [int(size * 0.70), int(size * 0.85)],
                )
                train_idx.extend(list(session_idx_train))
                val_idx.extend(list(session_idx_val))
                test_idx.extend(list(session_idx_test))
    elif args.split_mode == 2:
        # stratify on state such that segments from a given individual are
        # assigned to one split only i.e. individuals do not overlap across splits
        id_codes = np.unique(y[:, Y_COLS.index("NHC")])
        states = [
            y[y[:, Y_COLS.index("NHC")] == c][
                Y_COLS.index("NHC"), Y_COLS.index("status")
            ]
            for c in id_codes
        ]
        # too few mxs instances at the moment, treat mxs as me
        states = np.where(
            np.array(states) == float(DICT_STATE["mxs"]),
            DICT_STATE["me"],
            np.array(states),
        )
        idx = np.arange((len(states)))
        train_idx, test_idx, y_train, y_test = train_test_split(
            idx, states, stratify=states, train_size=0.7, random_state=123
        )
        val_idx, test_idx, y_val, y_test = train_test_split(
            test_idx, y_test, stratify=y_test, train_size=0.5, random_state=123
        )
        id_codes_train, id_codes_val, id_codes_test = (
            id_codes[train_idx],
            id_codes[val_idx],
            id_codes[test_idx],
        )
        train_idx, val_idx, test_idx = (
            [i for i, c in enumerate(y[:, Y_COLS.index("NHC")]) if c in id_codes_train],
            [i for i, c in enumerate(y[:, Y_COLS.index("NHC")]) if c in id_codes_val],
            [i for i, c in enumerate(y[:, Y_COLS.index("NHC")]) if c in id_codes_test],
        )
    else:
        raise NotImplementedError(
            f"split_mode {args.split_mode} has not been implemented."
        )

    return {
        "x_train": {c: r[train_idx] for c, r in x.items()},
        "y_train": y[train_idx],
        "x_val": {c: r[val_idx] for c, r in x.items()},
        "y_val": y[val_idx],
        "x_test": {c: r[test_idx] for c, r in x.items()},
        "y_test": y[test_idx],
    }


def preprocess(args):
    select_items(args)

    if args.verbose:
        print(f"\nLoading data from {args.dataset}...")
    clinical_info = read_clinical_info(os.path.join(FILE_DIRECTORY, "database.xlsx"))
    clinical_info.replace({"status": DICT_STATE}, inplace=True)
    clinical_info.replace({"time": DICT_TIME}, inplace=True)

    ds_info = {
        "label_scale": LABEL_SCALE,
        "time_alignment": args.time_alignment,
        "downsampling": args.downsampling,
        "padding_mode": args.padding_mode,
        "qc_mode": args.qc_mode,
        "segment_length": args.segment_length,
    }

    # features: dictionary of list of np.ndarray where the keys are the channel
    features, labels, sessions_info = {}, [], {}
    for session_id in tqdm(
        list(args.session_codes.values())[0],
        desc=f"Preprocessing",
        disable=args.verbose == 0,
    ):
        recording_dir = utils.unzip_session(args.dataset, session_id=session_id)
        session_data, session_label, session_info = preprocess_dir(
            args,
            recording_dir=recording_dir,
            clinical_info=clinical_info,
        )

        for info_name in ["channel_names", "channel_freq", "sampling_rates"]:
            if info_name not in ds_info:
                ds_info[info_name] = session_info[info_name]
            del session_info[info_name]
        sessions_info[session_id] = session_info

        session_data, num_segments = segmentation(
            args,
            session_data=session_data,
            channel_freq=ds_info["channel_freq"],
        )

        if num_segments:
            update_dict(features, session_data)
            # duplicate num_segments number of labels
            labels.extend([session_label for _ in range(num_segments)])
        else:
            print(f"Session {session_id} has no valid segments")
        del session_data, session_label, num_segments

    # joint features and labels from all sessions
    features = {c: np.concatenate(r, axis=0) for c, r in features.items()}
    labels = np.array(labels, dtype=np.float32)

    ds_info["sessions_info"] = sessions_info

    filter_data.set_unique_rec_id(args, y=labels)
    data = {"x": features, "y": labels}

    return data, ds_info
