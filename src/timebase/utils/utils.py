import os
import re
import csv
import copy
import platform
import subprocess
import typing as t
import numpy as np
import seaborn as sns
from glob import glob
import tensorflow as tf
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from timebase.utils import yaml, tensorboard, metrics
from timebase.diagnostic.registry import get_explainer

import matplotlib

if platform.system() == "Darwin":
    matplotlib.use("TkAgg")


def save_checkpoint(
    args, checkpoint: tf.train.Checkpoint, epoch: int, filename: str = None
):
    if filename is None:
        filename = os.path.join(args.checkpoint_dir, f"epoch-{epoch:03d}")
    else:
        filename = os.path.join(args.checkpoint_dir, filename)
    path = checkpoint.write(filename)
    if args.verbose == 2:
        print(f"saved checkpoint to {filename}\n")
    return path


def load_checkpoint(args, checkpoint: tf.train.Checkpoint, force: bool = False):
    """
    Load the best checkpoint or the latest checkpoint from args.checkpoint_dir
    if available, and return the epoch number of that checkpoint.
    Args:
      args
      checkpoint: tf.train.Checkpoint, TensorFlow Checkpoint object
      force: bool, raise an error if no checkpoint is found.
    Returns:
      epoch: int, the epoch number of the loaded checkpoint, 0 otherwise.
    """
    epoch, ckpt_filename = 0, None
    # load best model if exists, otherwise load the latest model if exists.
    best_model_yaml = os.path.join(args.checkpoint_dir, "best_model.yaml")
    if os.path.exists(best_model_yaml):
        best_model_info = yaml.load(best_model_yaml)
        epoch = best_model_info["epoch"]
        ckpt_filename = os.path.join(args.checkpoint_dir, best_model_info["path"])
    else:
        checkpoints = sorted(glob(os.path.join(args.checkpoint_dir, "*.index")))
        if checkpoints:
            ckpt_filename = checkpoints[-1].replace(".index", "")
    if force and not ckpt_filename:
        raise FileNotFoundError(f"no checkpoint found in {args.output_dir}.")
    if ckpt_filename:
        status = checkpoint.restore(ckpt_filename)
        status.expect_partial()
        if epoch == 0:
            match = re.match(r".+epoch-(\d{3})", ckpt_filename)
            epoch = int(match.groups()[0])
        if args.verbose:
            print(f"loaded checkpoint from {ckpt_filename}\n")
    return epoch


def update_dict(target: t.Dict, source: t.Dict, replace: bool = False):
    """add or update items in source to target"""
    for key, value in source.items():
        if replace:
            target[key] = value
        else:
            if key not in target:
                target[key] = []
            target[key].append(value)


def check_output(command: list):
    """Execute command in subprocess and return output as string"""
    return subprocess.check_output(command).strip().decode()


def save_args(args):
    """Save args object as dictionary to output_dir/args.yaml"""
    arguments = copy.deepcopy(args.__dict__)
    # remove session2class as yaml does not support tuple
    arguments.pop("session2class", None)
    arguments["git_hash"] = check_output(["git", "describe", "--always"])
    arguments["hostname"] = check_output(["hostname"])
    yaml.save(filename=os.path.join(args.output_dir, "args.yaml"), data=arguments)


def load_args(args, experiment):
    """Load args from output_dir/args.yaml"""
    filename = os.path.join(
        [f for f in glob(os.path.join(experiment, "*")) if f.endswith(args.algorithm)][
            0
        ],
        "args.yaml",
    )
    assert os.path.exists(filename)
    arguments = yaml.load(filename=filename)
    for key, value in arguments.items():
        if key not in [
            "class2name",
            "class2session",
            "session2class",
            "train_steps",
            "val_steps",
            "test_steps",
            "ds_info",
        ]:
            setattr(args, key, value)


def load_args_oos(args):
    """Load args from output_dir/args.yaml"""
    filename = os.path.join(args.output_dir, "args.yaml")
    assert os.path.exists(filename)
    arguments = yaml.load(filename=filename)
    for key, value in arguments.items():
        if not hasattr(args, key) and key not in [
            "class2name",
            "class2session",
            "session2class",
            "train_steps",
            "val_steps",
            "test_steps",
            "ds_info",
        ]:
            setattr(args, key, value)


def plot_confusion_matrix(
    ds, model, summary: tensorboard.Summary, epoch: int, mode: int
):
    y_true, y_pred = [], []
    for x, y in ds:
        y_true.append(y)
        y_pred.append(tf.argmax(model(x, training=False), axis=-1))
    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)
    summary.confusion_matrix(
        tag="confusion_matrix", y_true=y_true, y_pred=y_pred, step=epoch, mode=mode
    )


def accuracy(cm: np.ndarray):
    """Compute accuracy given Numpy array confusion matrix cm. Returns a
    floating point value"""
    return np.sum(np.diagonal(cm)) / np.sum(cm)


def save_confusion_matrix(
    args, filename: str, cm: np.ndarray, class_names: t.List[str]
):
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [6, 1], "wspace": 0.1},
        dpi=args.dpi,
    )
    sns.heatmap(
        cm / np.sum(cm, axis=-1)[:, np.newaxis],
        cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
        annot=cm.astype(str),
        fmt="",
        linewidths=0.01,
        cbar=True,
        cbar_ax=axes[1],
        ax=axes[0],
    )
    axes[0].set_xlabel("Prediction")
    axes[0].set_ylabel("Ground truth")
    axes[0].set_xticklabels(class_names, va="top", ha="center")
    axes[0].set_yticklabels(class_names, va="center", ha="right")
    axes[1].tick_params(axis="both", labelsize=9)
    figure.subplots_adjust(wspace=0.05, hspace=0.05)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    plt.savefig(filename, dpi=args.dpi)


def write_csv(output_dir, content: list):
    with open(os.path.join(output_dir, "results.csv"), "a") as file:
        writer = csv.writer(file)
        writer.writerow(content)
