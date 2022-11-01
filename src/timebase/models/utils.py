import io
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def Activation(activation: str, **kwargs):
    if activation in ["lrelu", "leakyrelu"]:
        return tf.keras.layers.LeakyReLU(**kwargs)
    else:
        return tf.keras.layers.Activation(activation, **kwargs)


def Normalization(normalization: str, **kwargs):
    if normalization in ["layer_norm", "layernorm"]:
        return tf.keras.layers.LayerNormalization(**kwargs)
    elif normalization in ["batch_norm", "batchnorm"]:
        return tf.keras.layers.BatchNormalization(**kwargs)
    elif normalization in ["instance_norm", "instancenorm"]:
        return tfa.layers.InstanceNormalization(**kwargs)
    elif normalization in ["group_norm", "groupnorm"]:
        return tfa.layers.GroupNormalization(**kwargs)
    raise NameError(f"Normalization layer {normalization} not found.")


def count_trainable_params(model):
    """Return the number of trainable parameters"""
    return np.sum([tf.keras.backend.count_params(p) for p in model.trainable_variables])


def model_summary(args, model):
    """Return tf.keras model summary as a string and save result to model.txt"""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary = stream.getvalue()
    stream.close()
    with open(os.path.join(args.output_dir, "model.txt"), "a") as file:
        file.write(summary)
    return summary


def regularize_parameters(model: tf.keras.Model, alpha: float, beta: float):
    loss = 0
    for layer in model.layers:
        for weight in layer.trainable_weights:
            if "bias" not in weight.name:
                reg_loss = alpha * (
                    (beta) * (tf.norm(weight, ord=1))
                    + (1 - beta) * (tf.norm(weight, ord=2))
                )
                loss += reg_loss
    return loss
