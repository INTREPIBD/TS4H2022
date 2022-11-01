import tensorflow as tf


def _reduce_losses(
    losses: tf.Tensor, reduction: tf.keras.losses.Reduction
) -> tf.Tensor:
    """Reduces losses to specified reduction."""
    if reduction == tf.keras.losses.Reduction.NONE:
        return losses
    elif reduction in [
        tf.keras.losses.Reduction.AUTO,
        tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
    ]:
        return tf.reduce_mean(losses)
    elif reduction == tf.keras.losses.Reduction.SUM:
        return tf.reduce_sum(losses)
    else:
        raise Exception(f"{reduction} is not a valid reduction.")


def cross_entropy(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_mean(
        tf.losses.sparse_categorical_crossentropy(
            y_true=y_true, y_pred=y_pred, from_logits=False
        )
    )


def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor):
    return tf.reduce_mean(
        tf.metrics.sparse_categorical_accuracy(y_true=y_true, y_pred=y_pred)
    )


def mae(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
):
    return _reduce_losses(tf.math.abs(tf.math.subtract(x=y_true, y=y_pred)), reduction)


def rmse(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
):
    return tf.math.sqrt(
        _reduce_losses(
            tf.math.square(tf.math.subtract(x=y_true, y=y_pred)), reduction=reduction
        )
    )


def mse(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
):
    return _reduce_losses(
        tf.math.square(tf.math.subtract(x=y_true, y=y_pred)), reduction
    )
