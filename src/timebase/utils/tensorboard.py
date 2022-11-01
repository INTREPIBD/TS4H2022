import os
import io
import platform
import matplotlib
import numpy as np
import typing as t
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt


sns.set_style("ticks")
plt.style.use("seaborn-deep")

PARAMS_PAD = 1
PARAMS_LENGTH = 2

plt.rcParams.update(
    {
        "mathtext.default": "regular",
        "xtick.major.pad": PARAMS_PAD,
        "ytick.major.pad": PARAMS_PAD,
        "xtick.major.size": PARAMS_LENGTH,
        "ytick.major.size": PARAMS_LENGTH,
        "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
        "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
        "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
    }
)

TICKER_FORMAT = matplotlib.ticker.FormatStrFormatter("%.2f")

JET = cm.get_cmap("jet")
GRAY = cm.get_cmap("gray")
TURBO = cm.get_cmap("turbo")
COLORMAP = TURBO
GRAY2RGB = COLORMAP(np.arange(256))[:, :3]


def remove_spines(axis: matplotlib.axes.Axes):
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)


def remove_top_right_spines(axis: matplotlib.axes.Axes):
    """Remove the ticks and spines of the top and right axis"""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def set_right_label(axis: matplotlib.axes.Axes, label: str, fontsize: int = None):
    """Set y-axis label on the right-hand side"""
    right_axis = axis.twinx()
    kwargs = {"rotation": 270, "va": "center", "labelpad": 3}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    right_axis.set_ylabel(label, **kwargs)
    right_axis.set_yticks([])
    remove_top_right_spines(right_axis)


def set_xticks(
    axis: matplotlib.axes.Axes,
    ticks_loc: t.Union[np.ndarray, list],
    ticks: t.Union[np.ndarray, list],
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
):
    axis.set_xticks(ticks_loc)
    axis.set_xticklabels(ticks, fontsize=tick_fontsize)
    if label:
        axis.set_xlabel(label, fontsize=label_fontsize)


def set_yticks(
    axis: matplotlib.axes.Axes,
    ticks_loc: t.Union[np.ndarray, list],
    ticks: t.Union[np.ndarray, list],
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
):
    axis.set_yticks(ticks_loc)
    axis.set_yticklabels(ticks, fontsize=tick_fontsize)
    if label:
        axis.set_ylabel(label, fontsize=label_fontsize)


def set_ticks_params(
    axis: matplotlib.axes.Axes, length: int = PARAMS_LENGTH, pad: int = PARAMS_PAD
):
    axis.tick_params(axis="both", which="both", length=length, pad=pad)


def save_figure(figure: plt.Figure, filename: str, dpi: int = 120, close: bool = True):
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    figure.savefig(
        filename, dpi=dpi, bbox_inches="tight", pad_inches=0.01, transparent=True
    )
    if close:
        plt.close(figure)


class Summary(object):
    """Helper class to write TensorBoard summaries"""

    def __init__(self, args, output_dir: str = ""):
        self.dpi = args.dpi
        self.format = args.format
        self.dataset = args.dataset
        self.save_plots = args.save_plots
        self.channel_names = args.ds_info["channel_names"]

        # write TensorBoard summary to specified output_dir or args.output_dir
        if output_dir:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            self.writers = [tf.summary.create_file_writer(output_dir)]
        else:
            output_dir = args.output_dir
            self.writers = [
                tf.summary.create_file_writer(output_dir),
                tf.summary.create_file_writer(os.path.join(output_dir, "val")),
                tf.summary.create_file_writer(os.path.join(output_dir, "test")),
            ]

        self.plots_dir = os.path.join(output_dir, "plots")
        if not os.path.isdir(self.plots_dir):
            os.makedirs(self.plots_dir)

        if platform.system() == "Darwin" and args.verbose == 2:
            matplotlib.use("TkAgg")

    def get_writer(self, mode: int = 0):
        return self.writers[mode]

    def close(self):
        for writer in self.writers:
            writer.close()

    def scalar(self, tag, value, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        with writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def histogram(self, tag, values, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        with writer.as_default():
            tf.summary.histogram(tag, values, step=step)

    def image(self, tag, values, step: int = 0, mode: int = 0):
        writer = self.get_writer(mode)
        with writer.as_default():
            tf.summary.image(tag, data=values, step=step, max_outputs=len(values))

    def figure(
        self,
        tag: str,
        figure: plt.Figure,
        step: int = 0,
        close: bool = True,
        mode: int = 0,
    ):
        """Write matplotlib figure to summary
        Args:
          tag: str, data identifier
          figure: plt.Figure, matplotlib figure or a list of figures
          step: int, global step value to record
          close: bool, close figure if True
          mode: int, indicate which summary writers to use
        """
        if self.save_plots:
            save_figure(
                figure,
                filename=os.path.join(
                    self.plots_dir, f"epoch_{step:03d}", f"{tag}.{self.format}"
                ),
                dpi=self.dpi,
                close=False,
            )
        buffer = io.BytesIO()
        figure.savefig(
            buffer, dpi=self.dpi, format="png", bbox_inches="tight", pad_inches=0.02
        )
        buffer.seek(0)
        image = tf.image.decode_png(buffer.getvalue(), channels=4)
        self.image(tag, tf.expand_dims(image, 0), step=step, mode=mode)
        if close:
            plt.close(figure)

    def confusion_matrix(
        self,
        tag: str,
        y_true: t.Union[tf.Tensor, np.ndarray],
        y_pred: t.Union[tf.Tensor, np.ndarray],
        step: int = 0,
        mode: int = 0,
    ):
        if tf.is_tensor(y_true):
            y_true = y_true.numpy()
        if tf.is_tensor(y_pred):
            y_pred = y_pred.numpy()

        labels = list(range(len(self.class2name)))
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=labels)

        label_fontsize, tick_fontsize = 12, 10
        figure, ax = plt.subplots(
            nrows=1,
            ncols=1,
            gridspec_kw={"wspace": 0.05, "hspace": 0.05},
            figsize=(5, 5),
            dpi=self.dpi,
        )

        cmap = sns.color_palette("rocket_r", as_cmap=True)
        sns.heatmap(
            confusion_matrix / np.sum(confusion_matrix, axis=-1)[:, np.newaxis],
            vmin=0,
            vmax=1,
            cmap=cmap,
            annot=confusion_matrix.astype(str),
            fmt="",
            linewidths=0.01,
            cbar=False,
            ax=ax,
        )

        ax.set_xlabel(f"Predictions", fontsize=label_fontsize)
        ax.set_ylabel("Targets", fontsize=label_fontsize)
        ax.set_title(
            f"Accuracy: {metrics.accuracy_score(y_true, y_pred):.04f}  | "
            f"F1-score: {metrics.f1_score(y_true, y_pred, labels=labels, average='macro'):.04f}",
            fontsize=tick_fontsize,
        )

        ticklabels = [self.class2name[i] for i in range(len(self.class2name))]
        ax.set_xticklabels(
            ticklabels, va="top", ha="center", rotation=90, fontsize=tick_fontsize
        )
        ax.set_yticklabels(
            ticklabels, va="center", ha="right", rotation=0, fontsize=tick_fontsize
        )
        set_ticks_params(axis=ax, length=0, pad=PARAMS_PAD + 2)

        # set color bar
        pos = ax.get_position()
        width, height = 0.015, pos.y1 * 0.3
        cbar_ax = figure.add_axes(
            rect=[
                pos.x1 + 0.02,
                ((pos.y1 - pos.y0) / 2 + pos.y0) - (height / 2),
                width,
                height,
            ]
        )
        figure.colorbar(cm.ScalarMappable(cmap="rocket_r"), cax=cbar_ax, shrink=0.1)
        ytick_loc = np.linspace(0, 1, 3)
        set_yticks(
            axis=cbar_ax,
            ticks_loc=ytick_loc,
            ticks=np.round(ytick_loc, 1),
            label="",
            tick_fontsize=tick_fontsize,
        )
        set_ticks_params(axis=cbar_ax)

        self.figure(tag, figure, step=step, mode=mode)

    def errors_distribution(
        self,
        args,
        tag: str,
        loss_scaled: np.ndarray,
        loss_unscaled: np.ndarray,
        mae: np.ndarray,
        rmse: np.ndarray,
        step: int = 0,
        mode: int = 0,
    ):

        label_fontsize, title_fontsize = 8, 16
        figure, axs = plt.subplots(
            nrows=4,
            ncols=1,
            gridspec_kw={"wspace": 0.02, "hspace": 0.50},
            figsize=(18, 10),
            dpi=self.dpi,
        )

        my_pal = {item: "r" for item in args.selected_items if item.startswith("YMRS")}
        my_pal.update(
            {item: "b" for item in args.selected_items if item.startswith("HDRS")}
        )
        titles = ["loss scaled", "loss unscaled", "mae", "rmse"]
        for idx, err in enumerate([loss_scaled, loss_unscaled, mae, rmse]):
            dict = {
                "Item": list(
                    np.repeat([item for item in args.selected_items], len(err))
                ),
                "Error": err.flatten(order="F"),
            }
            df = pd.DataFrame(dict)

            sns.barplot(data=df, x="Item", y="Error", ax=axs[idx], palette=my_pal)
            axs[idx].set_xticks([i for i in range(len(args.selected_items))])
            axs[idx].set_xticklabels(
                [i for i in args.selected_items], rotation=45, fontsize=label_fontsize
            )
            axs[idx].set_ylabel("")
            axs[idx].set_xlabel("")
            axs[idx].set_title(titles[idx], fontsize=title_fontsize)

        self.figure(tag, figure, step=step, mode=mode)

    def errors_distribution_regression(
        self,
        tag: str,
        loss: t.List,
        rmse: t.List,
        step: int = 0,
        mode: int = 0,
    ):

        label_fontsize, tick_fontsize = 12, 10
        figure, axs = plt.subplots(
            nrows=2,
            ncols=2,
            gridspec_kw={"wspace": 0.20, "hspace": 0.20},
            figsize=(13, 8),
            dpi=self.dpi,
        )

        scale_items = [f"YMRS_{i + 1}" for i in range(11)] + [
            f"HDRS_{i + 1}" for i in range(17)
        ]
        loss_dict = {scale_items[i]: loss[i] for i in range(len(scale_items))}
        rmse_dict = {scale_items[i]: rmse[i] for i in range(len(scale_items))}
        scales, scores = ["YMRS", "HDRS"], [loss_dict, rmse_dict]
        color = ["red", "blue"]
        names = ["loss", "rmse"]

        for c_idx, scale in enumerate(scales):
            for r_idx, score in enumerate(scores):
                height = [v for k, v in score.items() if k.startswith(scale)]
                axs[r_idx, c_idx].bar(
                    [i for i in range(len(height))],
                    height,
                    color=color[scales.index(scale)],
                )
                if not r_idx:
                    axs[r_idx, c_idx].set_title(
                        f"{scale}",
                        fontsize=tick_fontsize,
                    )

                axs[r_idx, c_idx].set_xticks([i for i in range(len(height))])
                axs[r_idx, c_idx].set_xticklabels(
                    [i + 1 for i in range(len(height))], rotation=45
                )
                axs[r_idx, c_idx].set_ylabel(f"{names[r_idx]}", fontsize=label_fontsize)

        plt.xticks(rotation=45, ha="right")

        self.figure(tag, figure, step=step, mode=mode)

    def plot_diagnostic(
        self,
        args,
        tag: str,
        items_L2_norm=np.ndarray,
        cosine_matrix=np.ndarray,
        step: int = 0,
        mode: int = 0,
    ):

        figure = plt.figure(figsize=(15, 15), dpi=self.dpi)
        gs = figure.add_gridspec(nrows=2, ncols=2, wspace=0.20, hspace=0.20)
        ax1 = figure.add_subplot(gs[0, :])
        ax2 = figure.add_subplot(gs[1, 0])
        ax3 = figure.add_subplot(gs[1, 1])

        label_fontsize, title_fontsize = 10, 12

        # ax1
        box_dict = {
            "Item": list(
                np.repeat([item for item in args.selected_items], len(items_L2_norm))
            ),
            "Grad": items_L2_norm.flatten(order="F"),
        }
        box_df = pd.DataFrame(box_dict)

        my_pal = {item: "r" for item in args.selected_items if item.startswith("YMRS")}
        my_pal.update(
            {item: "b" for item in args.selected_items if item.startswith("HDRS")}
        )
        sns.boxplot(data=box_df, x="Item", y="Grad", ax=ax1, palette=my_pal)
        ax1.set_xticks([i for i in range(len(args.selected_items))])
        ax1.set_xticklabels(
            [i for i in args.selected_items], rotation=45, fontsize=label_fontsize
        )
        ax1.set_title("Items Gradient - L2 norm", fontsize=title_fontsize)

        # ax2, ax3
        mean_cosine = np.mean(cosine_matrix, axis=0)
        std_cosine = np.std(cosine_matrix, axis=0)
        mask = np.triu(np.ones_like(mean_cosine, dtype=bool))
        cmap = sns.color_palette("rocket_r", as_cmap=True)

        titles = [
            "Cosine mean",
            "Cosine std",
        ]
        for idx, (matrix, ax) in enumerate(zip([mean_cosine, std_cosine], [ax2, ax3])):
            sns.heatmap(
                matrix,
                annot=np.round(matrix, 2).astype(str),
                mask=mask,
                cmap=cmap,
                fmt="",
                square=True,
                linewidths=0.01,
                vmin=0,
                vmax=1,
                ax=ax,
                cbar=False,
            )
            ax.set_xticks([i for i in range(len(args.selected_items))])
            ax.set_xticklabels(
                args.selected_items,
                va="top",
                ha="center",
                rotation=90,
                fontsize=label_fontsize,
            )
            ax.set_yticklabels(
                args.selected_items,
                va="center",
                ha="right",
                rotation=0,
                fontsize=label_fontsize,
            )
            ax.set_title(titles[idx], fontsize=title_fontsize)

        self.figure(tag, figure, step=step, mode=mode)

    def plot_samples(
        self,
        ds: tf.data.Dataset,
        epoch: int,
        mode: int,
        num_samples: int = 5,
    ):
        """Plot the first sample in num_samples batches in dataset ds"""
        for i, (x, y, _) in enumerate(ds.take(num_samples)):
            nrows = len(x)
            figure, axes = plt.subplots(
                nrows=nrows,
                ncols=1,
                gridspec_kw={"wspace": 0.1, "hspace": 0.3},
                figsize=(4.5, 7.5),
                dpi=self.dpi,
            )
            for i, (channel, recording) in enumerate(x.items()):
                axes[i].plot(recording.numpy()[0, :], linewidth=1.5)
                remove_top_right_spines(axis=axes[i])
                set_right_label(axis=axes[i], label=self.channel_names[i])
            axes[-1].set_xlabel("Time-step")

            self.figure(tag=f"samples/{i:03d}", figure=figure, step=epoch, mode=mode)

    def plot_regression_errors(
        self,
        y_true: t.Union[tf.Tensor, np.ndarray],
        y_pred: t.Union[tf.Tensor, np.ndarray],
        epoch: int,
        mode: int,
    ):
        assert y_true.shape == y_pred.shape
        if isinstance(y_true, tf.Tensor):
            y_true = y_true.numpy()
        if isinstance(y_pred, tf.Tensor):
            y_pred = y_pred.numpy()

        figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(8, 2.5), dpi=self.dpi)

        tick_fontsize, label_fontsize = 8, 11
        width, gap, align = 0.2, 0.05, "center"
        error_kw = {"linewidth": 1}

        diff = y_pred - y_true
        diff_mean, diff_std = np.mean(diff, axis=0), np.std(diff, axis=0)

        xticks = []

        # plot YMRS
        x = 0
        for i in range(11):
            xticks.append(x)
            axis.bar(
                x=x,
                height=diff_mean[i],
                width=width,
                align=align,
                color="orangered",
                edgecolor="orangered",
                label="YMRS" if i == 0 else "",
                yerr=diff_std[i],
                error_kw=error_kw,
            )
            if i != 10:
                x += width + gap

        x += width
        mid_point = x
        x += width

        # plot HDRS
        for i in range(11, 28):
            xticks.append(x)
            axis.bar(
                x=x,
                height=diff_mean[i],
                width=width,
                align=align,
                color="dodgerblue",
                edgecolor="dodgerblue",
                label="HDRS" if i == 11 else "",
                yerr=diff_std[i],
                error_kw=error_kw,
            )
            if i != 27:
                x += width + gap

        axis.legend(
            loc="upper right",
            ncol=2,
            frameon=False,
            handletextpad=0.35,
            handlelength=0.6,
            markerscale=0.8,
            fontsize=label_fontsize,
        )

        axis.hlines(y=0, xmin=-width, xmax=x + width, color="black", linewidth=0.8)
        axis.set_xlim(left=-width, right=x + width)

        ymin, ymax = axis.get_ylim()
        axis.vlines(
            x=mid_point,
            ymin=ymin,
            ymax=ymax,
            colors="black",
            linestyle="--",
            linewidth=1,
        )
        axis.set_ylim(bottom=np.floor(ymin), top=np.ceil(ymax))

        yticks_loc = axis.get_yticks()
        set_yticks(
            axis=axis,
            ticks_loc=yticks_loc,
            ticks=yticks_loc,
            label="Residual",
            tick_fontsize=tick_fontsize,
            label_fontsize=label_fontsize,
        )

        set_xticks(
            axis=axis,
            ticks_loc=xticks,
            ticks=list(range(1, 12)) + list(range(1, 18)),
            label="Item",
            tick_fontsize=tick_fontsize + 1,
            label_fontsize=label_fontsize,
        )
        axis.tick_params(axis="x", which="both", length=0)
        axis.tick_params(axis="y", which="both", length=3)

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(False)

        self.figure(tag="item_residual", figure=figure, step=epoch, mode=mode)
