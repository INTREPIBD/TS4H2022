# All the static variables related to the dataset

import os
import numpy as np

FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

HR_OFFSET = 10  # HR data record 10s after the t0
# channel names of the csv recorded data
CSV_CHANNELS = ["ACC", "BVP", "EDA", "HR", "TEMP", "IBI"]
# HRV: time domain features
HRV_FEATURES = [
    "mean_nni",
    "sdnn",
    "sdsd",
    "nni_50",
    "pnni_50",
    "nni_20",
    "pnni_20",
    "rmssd",
    "median_nni",
    "range_nni",
    "cvsd",
    "cvnni",
    "mean_hr",
    "max_hr",
    "min_hr",
    "std_hr",
]

# maximum values of each individual symptom in YMRS and HDRS
MAX_YMRS = [4, 4, 4, 4, 8, 8, 4, 8, 8, 4, 4]
MAX_HDRS = [4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 2, 2, 2, 4, 3, 2]
# MAX_HDRS = {1:4, 2:4, 3:4, 4:2, 5:2, 6:2, 7:4, 8:4, 9:4, 10:4, 11:4, 12:2, 13:2, 14:2, 15:4, 16:3, 17:2}
# label format [session ID, is patient, timing, YMRS(1 - 11), HDRS(1 - 17)]
LABEL_SCALE = np.array([1, 1, 1] + MAX_YMRS + MAX_HDRS, dtype=np.float32)

YMRS_item_ranks = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
HDRS_item_ranks = [5, 5, 5, 3, 3, 3, 5, 5, 5, 5, 5, 3, 3, 3, 5, 4, 3]

# some items of the YMRS are scored with 2 unit interval spaces
RANK_NORMALIZER = [
    1,
    1,
    1,
    1,
    2,
    2,
    1,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]

DICT_STATE = {
    "MDE_BD": 0,
    "MDE_MDD": 1,
    "ME": 2,
    "MX": 3,
    "PE": 4,
    "Eu_BD": 5,
    "Eu_MDD": 6,
    "SP": 7,
    "HC": 8,
    "INC": 8,
    "SUD": 9,
}
DICT_TIME = {"T0": 0, "T1": 1, "T2": 2, "T3": 3, "T4": 4}

Y_COLS = [
    "NHC",
    "age",
    "sex",
    "status",
    "time",
    "Session_Code",
    "YMRS1",
    "YMRS2",
    "YMRS3",
    "YMRS4",
    "YMRS5",
    "YMRS6",
    "YMRS7",
    "YMRS8",
    "YMRS9",
    "YMRS10",
    "YMRS11",
    "YMRS_SUM",
    "HDRS1",
    "HDRS2",
    "HDRS3",
    "HDRS4",
    "HDRS5",
    "HDRS6",
    "HDRS7",
    "HDRS8",
    "HDRS9",
    "HDRS10",
    "HDRS11",
    "HDRS12",
    "HDRS13",
    "HDRS14",
    "HDRS15",
    "HDRS16",
    "HDRS17",
    "HDRS_SUM",
    "IPAQ_total",
    "YMRS_discretized",
    "HDRS_discretized",
]
