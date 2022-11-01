import os
import numpy as np
import typing as t
import pandas as pd
from math import ceil
import tensorflow as tf
from copy import deepcopy

from timebase.utils import yaml
from timebase.data import preprocessing, utils
from timebase.data.static import Y_COLS

AUTOTUNE = tf.data.AUTOTUNE


def dry_run(datasets: t.List[tf.data.Dataset]):
    count = 0
    for ds in datasets:
        for _ in ds:
            count += 1


def compute_statistics(
    args, data: t.Dict[str, t.Union[np.ndarray, t.Dict[str, np.ndarray]]]
):
    """
    Compute the min, max, mean and standard deviation of the training,
    validation and test sets
    """
    ds_names = ["x_train", "x_val", "x_test"]
    channels = list(args.input_shapes.keys())
    # compute min, max, mean and std of the features in each set
    stats = {ds_name: {} for ds_name in ds_names}
    for ds_name in ds_names:
        for channel in channels:
            channel_data = data[ds_name][channel]
            stats[ds_name][channel] = {
                "min": np.min(channel_data),
                "max": np.max(channel_data),
                "mean": np.mean(channel_data),
                "std": np.std(channel_data),
            }
    return stats


def scale_features(
    args, data: t.Dict[str, t.Union[np.ndarray, t.Dict[str, np.ndarray]]]
):
    """scale features according to args.scaling_mode:
    - 0: no scaling
    - 1: normalization (lean parameters from train and apply across splits)
    - 2: standardisation (lean parameters from train and apply across splits)
    """
    if args.scaling_mode == 0:
        pass
    elif args.scaling_mode == 1:
        stats = args.ds_info["stats"]["x_train"]
        for ds_name in ["x_train", "x_val", "x_test"]:
            for channel, recordings in data[ds_name].items():
                data[ds_name][channel] = utils.normalize(
                    recordings,
                    x_min=stats[channel]["min"],
                    x_max=stats[channel]["max"],
                )
    elif args.scaling_mode == 2:
        stats = args.ds_info["stats"]["x_train"]
        for ds_name in ["x_train", "x_val", "x_test"]:
            for channel, recordings in data[ds_name].items():
                data[ds_name][channel] = (recordings - stats[channel]["mean"]) / stats[
                    channel
                ]["std"]
    else:
        raise NotImplementedError(
            f"normalization mode {args.norm_mode} not implemented."
        )
    return data


def construct_dataset(
    args,
    data: t.Dict[str, t.Union[np.ndarray, t.Dict[str, np.ndarray]]],
):
    """Construct feature-label pairs for the specified regression_mode mode"""
    assert args.regression_mode in [0, 1]
    # select the labels that correspond to YMRS and HDRS
    item_idx = [Y_COLS.index(i) for i in args.selected_items]
    labels = data["y"][:, item_idx]

    if args.regression_mode == 0:
        labels = labels / args.item_max.numpy()

    # compute total score
    total_scores = np.sum(labels, axis=-1)
    # bin the index into 4 groups according to the total scores
    scores_idx = np.argsort(total_scores)
    groups = np.array_split(scores_idx, indices_or_sections=4)
    # sample 400 segments from each group for validation and test sets
    idx = {"train": [], "val": [], "test": []}
    for group in groups:
        size = len(group)
        group = np.random.permutation(group)
        set_idx = np.array_split(group, indices_or_sections=[size - 800, size - 400])
        idx["train"].append(set_idx[0])
        idx["val"].append(set_idx[1])
        idx["test"].append(set_idx[2])
    idx = {k: np.concatenate(v) for k, v in idx.items()}
    data["x_train"] = {k: v[idx["train"]] for k, v in data["x"].items()}
    data["y_train"] = labels[idx["train"]]

    data["x_val"] = {k: v[idx["val"]] for k, v in data["x"].items()}
    data["y_val"] = labels[idx["val"]]

    data["x_test"] = {k: v[idx["test"]] for k, v in data["x"].items()}
    data["y_test"] = labels[idx["test"]]

    del data["x"], data["y"]


def get_cat_weights(data: t.Dict[str, t.Union[np.ndarray, t.Dict[str, np.ndarray]]]):
    """
    get weights of each category to use in loss scaling
    """
    w_train, w_val, w_test = (
        np.zeros_like(data["y_train"]),
        np.zeros_like(data["y_val"]),
        np.zeros_like(data["y_test"]),
    )
    for item in range(w_train.shape[1]):
        ranks, counts = np.unique(data["y_train"][:, item], return_counts=True)
        item_proportion = {
            r: (c / len(data["y_train"])) ** -1 for r, c in zip(ranks, counts)
        }
        # normalize proportions
        item_proportion = {
            k: v / max(item_proportion.values()) for k, v in item_proportion.items()
        }
        w_train[:, item] = np.array(
            pd.Series(data["y_train"][:, item]).map(item_proportion)
        )
        w_val[:, item] = np.array(
            pd.Series(data["y_val"][:, item]).map(item_proportion)
        )
        w_test[:, item] = np.array(
            pd.Series(data["y_test"][:, item]).map(item_proportion)
        )

    assert (
        (np.isnan(w_train).any() == False)
        and (np.isnan(w_train).any() == False)
        and (np.isnan(w_train).any() == False)
    ), "item categories do not have the same support across train, val, test"
    data["w_train"], data["w_val"], data["w_test"] = w_train, w_val, w_test


class DataGenerator:
    def __init__(
        self,
        data: t.Dict[str, t.Union[np.ndarray, t.Dict[str, np.ndarray]]],
        ds_names: t.List[str],
        shuffle: bool = False,
    ):
        assert len(ds_names) == 3
        self._x = deepcopy(data[ds_names[0]])
        self._y = deepcopy(data[ds_names[1]])
        self._w = deepcopy(data[ds_names[2]])
        del data[ds_names[0]], data[ds_names[1]], data[ds_names[2]]
        self._shuffle = shuffle
        self._indexes = np.arange(self.__len__())
        if shuffle:
            self.shuffle_indexes()

    @property
    def signatures(self):
        return (
            {
                k: tf.TensorSpec(shape=v.shape[1:], dtype=tf.float32, name=k)
                for k, v in self._x.items()
            },
            tf.TensorSpec(shape=self._y.shape[1:], dtype=tf.float32, name="y"),
            tf.TensorSpec(shape=self._w.shape[1:], dtype=tf.float32, name="w"),
        )

    def __len__(self):
        return self._y.shape[0]

    def shuffle_indexes(self):
        self._indexes = np.random.permutation(self.__len__())

    def __getitem__(self, idx: int):
        x = {
            k: tf.convert_to_tensor(v[idx], dtype=tf.float32)
            for k, v in self._x.items()
        }
        y = tf.convert_to_tensor(self._y[idx], dtype=tf.float32)
        w = tf.convert_to_tensor(self._w[idx], dtype=tf.float32)
        return x, y, w

    def __call__(self):
        for i in range(self.__len__()):
            idx = self._indexes[i]
            yield self.__getitem__(idx=idx)

            if i == self.__len__() - 1 and self._shuffle:
                self.shuffle_indexes()


def get_datasets(args, buffer_size: int = 512):
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"dataset {args.dataset} not found.")

    args.session_codes = yaml.load(args.config)
    data, args.ds_info = preprocessing.preprocess(args)

    args.input_shapes = {c: r.shape[1:] for c, r in data["x"].items()}

    construct_dataset(args, data=data)
    args.ds_info["stats"] = compute_statistics(args, data=data)
    get_cat_weights(data=data)

    scale_features(args, data=data)

    args.train_steps = ceil(len(data["y_train"]) / args.batch_size)
    args.val_steps = ceil(len(data["y_val"]) / args.batch_size)
    args.test_steps = ceil(len(data["y_test"]) / args.batch_size)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (data["x_train"], data["y_train"], data["w_train"])
    )
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size)
    train_ds = train_ds.batch(args.batch_size)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (data["x_val"], data["y_val"], data["w_val"])
    )
    val_ds = val_ds.shuffle(buffer_size)
    val_ds = val_ds.batch(args.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (data["x_test"], data["y_test"], data["w_val"])
    )
    test_ds = test_ds.batch(args.batch_size)

    dry_run(datasets=[train_ds, val_ds, test_ds])

    return train_ds, val_ds, test_ds


def scramble_test_ds(args, x_test, y_test, to_permute: t.List = None):
    features = x_test.copy()
    if to_permute is not None:
        if len(to_permute) > 1:
            start, end = to_permute[0], to_permute[-1]
            np.random.seed(1234)
            np.random.shuffle(features[:, :, start:end])
        else:
            np.random.seed(1234)
            np.random.shuffle(features[:, :, to_permute[0]])
    test_ds = tf.data.Dataset.from_tensor_slices((features, y_test))
    test_ds = test_ds.batch(args.batch_size)

    dry_run(datasets=[test_ds])

    return test_ds
