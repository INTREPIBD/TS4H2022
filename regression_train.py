import os
import shutil
import pickle
import argparse
from tqdm import tqdm
from time import time
import tensorflow as tf

from timebase.data.reader import get_datasets
from timebase.utils.optimizer import Optimizer
from timebase.models.registry import get_model
from timebase.models.utils import regularize_parameters
from timebase.utils.early_stopping import EarlyStopping
from timebase.utils import tensorboard, utils, yaml, metrics


def check_positive_real(value):
    if float(value) < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive real value" % value)
    return float(value)


def check_zero_one(value):
    if 0 < float(value) < 1:
        raise argparse.ArgumentTypeError(f"{value} is not within (0, 1)")
    return float(value)


def predict(args, ds: tf.data.Dataset, model: tf.keras.Model, training: bool = False):
    y_true, y_pred = [], []
    item_max = args.item_max
    for x, y, _ in ds:
        prediction = model(x, training=training)
        if args.regression_mode == 0:
            y = y * item_max
            prediction = prediction * item_max
        y_true.append(y)
        y_pred.append(prediction)
    return tf.concat(y_true, axis=0).numpy(), tf.concat(y_pred, axis=0).numpy()


def make_plots(
    args,
    ds: tf.data.Dataset,
    model: tf.keras.Model,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
    save_predictions: bool = False,
):
    summary.plot_samples(ds=ds, epoch=epoch, mode=mode)
    y_true, y_pred = predict(args, ds=ds, model=model, training=False)
    if save_predictions:
        with open(
            os.path.join(args.output_dir, f"pred_epoch{epoch:03d}.pkl"), "wb"
        ) as file:
            pickle.dump({"y_true": y_true.numpy(), "y_pred": y_pred.numpy()}, file)
    summary.plot_regression_errors(
        y_true=y_true,
        y_pred=y_pred,
        epoch=epoch,
        mode=mode,
    )


def compute_metrics(y_true: tf.Tensor, y_pred: tf.Tensor):
    true_ymrs, pred_ymrs = y_true[:, :11], y_pred[:, :11]
    true_hdrs, pred_hdrs = y_true[:, 11:], y_pred[:, 11:]
    return {
        "metrics/MAE": metrics.mae(y_true=y_true, y_pred=y_pred),
        "metrics/RMSE": metrics.rmse(y_true=y_true, y_pred=y_pred),
        "metrics/RMSE_YMRS": metrics.rmse(y_true=true_ymrs, y_pred=pred_ymrs),
        "metrics/RMSE_HDRS": metrics.rmse(y_true=true_hdrs, y_pred=pred_hdrs),
        "metrics/RMSE_sum": metrics.rmse(
            y_true=tf.reduce_sum(y_true, axis=-1),
            y_pred=tf.reduce_sum(y_pred, axis=-1),
        ),
        "metrics/RMSE_YMRS_sum": metrics.rmse(
            y_true=tf.reduce_sum(true_ymrs, axis=-1),
            y_pred=tf.reduce_sum(pred_ymrs, axis=-1),
        ),
        "metrics/RMSE_HDRS_sum": metrics.rmse(
            y_true=tf.reduce_sum(true_hdrs, axis=-1),
            y_pred=tf.reduce_sum(pred_hdrs, axis=-1),
        ),
    }


@tf.function
def train_step(
    x: tf.Tensor,
    y: tf.Tensor,
    w: tf.Tensor,
    model: tf.keras.Model,
    optimizer: Optimizer,
    item_max: tf.Tensor,
    reg_alpha: float,
    reg_beta: float,
    item_delta: int,
    regression_mode: int,
):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = metrics.mse(
            y_true=y, y_pred=y_pred, reduction=tf.keras.losses.Reduction.NONE
        )
        if item_delta:
            loss = tf.math.multiply(loss, w)
        loss = tf.reduce_sum(loss, axis=-1)
        reg_loss = regularize_parameters(model=model, alpha=reg_alpha, beta=reg_beta)
        total_loss = loss + reg_loss
    optimizer.minimize(loss=total_loss, tape=tape)

    # scale labels and predictions into original range if needed
    if regression_mode == 0:
        y = y * item_max
        y_pred = y_pred * item_max
    result = {
        "loss/loss": loss,
        "loss/reg_loss": reg_loss,
        "loss/total_loss": total_loss,
    }
    result.update(compute_metrics(y_true=y, y_pred=y_pred))
    return result


def train(
    args,
    ds: tf.data.Dataset,
    model: tf.keras.Model,
    optimizer: Optimizer,
    summary: tensorboard.Summary,
    epoch: int,
):
    results = {}
    for x, y, w in tqdm(
        ds, desc="Train", total=args.train_steps, disable=args.verbose == 0
    ):
        result = train_step(
            x=x,
            y=y,
            w=w,
            model=model,
            optimizer=optimizer,
            item_max=args.item_max,
            reg_alpha=args.reg_alpha,
            reg_beta=args.reg_beta,
            item_delta=args.item_delta,
            regression_mode=args.regression_mode,
        )
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = tf.reduce_mean(tf.concat(v, axis=0)).numpy()
        summary.scalar(k, value=results[k], step=epoch, mode=0)
    return results


@tf.function
def validation_step(
    x: tf.Tensor,
    y: tf.Tensor,
    w: tf.Tensor,
    model: tf.keras.Model,
    item_max: tf.Tensor,
    reg_alpha: float,
    reg_beta: float,
    item_delta: int,
    regression_mode: int,
):
    y_pred = model(x, training=False)
    loss = metrics.mse(
        y_true=y, y_pred=y_pred, reduction=tf.keras.losses.Reduction.NONE
    )
    if item_delta:
        weights = tf.pow(w, item_delta)
        loss = tf.math.multiply(loss, weights)
    loss = tf.reduce_sum(loss, axis=-1)
    reg_loss = regularize_parameters(model=model, alpha=reg_alpha, beta=reg_beta)
    total_loss = loss + reg_loss

    # scale labels and predictions into original range if needed
    if regression_mode == 0:
        y = y * item_max
        y_pred = y_pred * item_max
    result = {
        "loss/loss": loss,
        "loss/reg_loss": reg_loss,
        "loss/total_loss": total_loss,
    }
    result.update(compute_metrics(y_true=y, y_pred=y_pred))
    return result


def validate(
    args,
    ds: tf.data.Dataset,
    model: tf.keras.Model,
    summary: tensorboard.Summary,
    epoch: int,
    mode: int = 1,
):
    results = {}
    for x, y, w in tqdm(
        ds, desc="Validation", total=args.val_steps, disable=args.verbose == 0
    ):
        result = validation_step(
            x=x,
            y=y,
            w=w,
            model=model,
            item_max=args.item_max,
            reg_alpha=args.reg_alpha,
            reg_beta=args.reg_beta,
            item_delta=args.item_delta,
            regression_mode=args.regression_mode,
        )
        utils.update_dict(target=results, source=result)
    for k, v in results.items():
        results[k] = tf.reduce_mean(tf.concat(v, axis=0)).numpy()
        summary.scalar(k, value=results[k], step=epoch, mode=mode)
    return results


def main(args):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)

    if args.clear_output_dir and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mixed_precision:
        if args.verbose:
            print(f"Enable mixed precision training.")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_ds, val_ds, test_ds = get_datasets(args)

    summary = tensorboard.Summary(args)

    model = get_model(args, summary)

    optimizer = Optimizer(args, model=model)

    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer.optimizer)

    early_stopping = EarlyStopping(args, model=model, checkpoint=checkpoint)
    epoch = utils.load_checkpoint(args, checkpoint=checkpoint)

    utils.save_args(args)
    make_plots(args, ds=val_ds, model=model, summary=summary, epoch=epoch, mode=1)

    results = {}
    while (epoch := epoch + 1) < args.epochs + 1:
        print(f"Epoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_results = train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            summary=summary,
            epoch=epoch,
        )
        val_results = validate(
            args, ds=val_ds, model=model, summary=summary, epoch=epoch
        )
        elapse = time() - start

        summary.scalar("elapse", value=elapse, step=epoch, mode=0)
        print(
            f'Train\t\ttotal_loss: {train_results["loss/total_loss"]:.04f}\t'
            f'RMSE: {train_results["metrics/RMSE_sum"]:.02f}\t'
            f'YMRS: {train_results["metrics/RMSE_YMRS_sum"]:.02f}\t'
            f'HDRS: {train_results["metrics/RMSE_HDRS_sum"]:.02f}\n'
            f'Validation\ttotal_loss: {val_results["loss/total_loss"]:.04f}\t'
            f'RMSE: {val_results["metrics/RMSE_sum"]:.02f}\t'
            f'YMRS: {val_results["metrics/RMSE_YMRS_sum"]:.02f}\t'
            f'HDRS: {val_results["metrics/RMSE_HDRS_sum"]:.02f}\n'
            f"Elapse: {elapse:.02f}s\n"
        )

        results.update({"train": train_results, "validation": val_results})
        if early_stopping.monitor(loss=val_results["loss/total_loss"], epoch=epoch):
            break
        if epoch % 20 == 0 or epoch == args.epochs:
            make_plots(
                args,
                ds=val_ds,
                model=model,
                summary=summary,
                epoch=epoch,
                mode=1,
            )

    early_stopping.restore()

    test_results = validate(
        args, ds=test_ds, model=model, summary=summary, epoch=epoch, mode=2
    )
    results.update({"test": test_results})
    make_plots(args, ds=test_ds, model=model, summary=summary, epoch=epoch, mode=2)

    with open(os.path.join(args.output_dir, "save.pkl"), "wb") as file:
        pickle.dump(
            {
                "train": predict(args, ds=train_ds, model=model, training=False),
                "val": predict(args, ds=val_ds, model=model, training=False),
                "test": predict(args, ds=test_ds, model=model, training=False),
            },
            file,
        )

    yaml.save(os.path.join(args.output_dir, "results.yaml"), data=results)

    print(f"Results saved to {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training configuration
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mixed_precision", action="store_true")

    # dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/raw_data",
        help="path to directory with raw data in zip files",
    )
    parser.add_argument(
        "--regression_mode",
        type=int,
        default=0,
        choices=[0, 1],
        help="regression mode: "
        "2) regression (normalized items), "
        "3) regression (unnormalized items)",
    )
    parser.add_argument(
        "--hdrs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        help="HDRS items: "
        "0 drop all HDRS items from target"
        "[1:17] item(s) to be included in target",
    )
    parser.add_argument(
        "--ymrs",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        help="YMRS items: "
        "0 drop all YMRS items from target"
        "[1:11] item(s) to be included in target",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config01.yaml",
        help="path to yaml file that contains the session code of the recordings",
    )
    parser.add_argument(
        "--time_alignment",
        type=int,
        default=0,
        choices=[0, 1, 2, 4, 8, 16, 32, 64],
        help="number of samples per second (Hz) for time-alignment, "
        "set 0 to train embedding layers instead.",
    )
    parser.add_argument(
        "--downsampling",
        type=str,
        default="average",
        choices=["average", "max"],
        help="downsampling method to use when --time_alignment > 0",
    )
    parser.add_argument(
        "--scaling_mode",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="normalize features: "
        "0) no normalization "
        "1) normalize features by the overall min and max values from the training set"
        "2) standardize features by the overall mean and standard deviation from the training set",
    )
    parser.add_argument(
        "--padding_mode",
        type=str,
        default="average",
        choices=["zero", "last", "average", "median"],
        help="padding mode for channels samples at a lower frequency",
    )
    parser.add_argument(
        "--qc_mode",
        type=int,
        default=1,
        choices=[0, 1],
        help="quality control mode:"
        "0 - no QC"
        "1 - 5-rules QC based on Kleckner et al. 2018",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=64,
        help="segmentation window length in seconds",
    )
    parser.add_argument(
        "--downsample_mode",
        type=int,
        default=1,
        choices=[0, 1],
        help="0) no downsampling, 1) downsample segments from majority class",
    )
    parser.add_argument(
        "--split_mode",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="criterion for train/val/test split:"
        "0) random splits"
        "1) split each session into 70:15:15 along the temporal dimension to "
        "minimize overlap between segments in train/val/test"
        "2) stratify on state such that a given participant is contained in one "
        "set only",
    )
    parser.add_argument(
        "--item_delta",
        type=int,
        default=0,
        choices=[0, 1],
        help="scale loss by the inverse of the item category ratio in training set,"
        "0 to disable.",
    )

    # embedding configuration
    parser.add_argument(
        "--embedding_type",
        type=int,
        default=1,
        choices=[0, 1],
        help="embedding to be used when args.time_alignment == 0"
        "0) GRU layer"
        "1) MLP layer",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="embedding dimension for each channel when args.time_alignment == 0",
    )

    # model configuration
    parser.add_argument(
        "--model", type=str, default="bilstm", choices=["mlp", "bilstm", "transformer"]
    )
    parser.add_argument("--num_units", type=int, default=128)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--diagnostic", type=str, default="items_gradient_detector")
    parser.add_argument("--extracted_features_layer", type=str, required=False)
    parser.add_argument(
        "--reg_alpha",
        type=check_positive_real,
        default=0.0001,
        help="coefficient to scales the regularisation loss",
    )
    parser.add_argument(
        "--reg_beta",
        type=check_zero_one,
        default=0.5,
        help="coefficient to controls the trade-off between L1 and L2 norm",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    # RNNs configuration
    parser.add_argument(
        "--r_dropout", type=float, default=0.0, help="Recurrent dropout in RNNs."
    )

    # Transformer configuration
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--head_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=32)
    parser.add_argument(
        "--t_dropout",
        type=float,
        default=0.25,
        help="Dropout rate in Transformer block.",
    )

    # matplotlib
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument(
        "--format", type=str, default="svg", choices=["pdf", "png", "svg"]
    )
    parser.add_argument("--dpi", type=int, default=120)

    # misc
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--clear_output_dir", action="store_true")

    params = parser.parse_args()
    main(params)
