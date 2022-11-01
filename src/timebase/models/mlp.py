from .registry import register

import tensorflow as tf

from timebase.models import embedding
from timebase.models.utils import Activation


@register("mlp")
def get_model(args, name: str = "mlp"):
    inputs, embeddings = embedding.encoder(args)

    outputs = tf.keras.layers.Flatten(name="flatten")(embeddings)

    outputs = tf.keras.layers.Dense(units=args.num_units, name="dense1")(outputs)
    outputs = tf.keras.layers.Dropout(rate=args.dropout, name="dropout1")(outputs)
    outputs = Activation(activation=args.activation, name="activation1")(outputs)

    outputs = tf.keras.layers.Dense(units=args.num_units // 2, name="dense2")(outputs)
    outputs = tf.keras.layers.Dropout(rate=args.dropout, name="dropout2")(outputs)
    outputs = Activation(activation=args.activation, name="activation2")(outputs)

    outputs = tf.keras.layers.Dense(units=args.num_units // 3, name="dense3")(outputs)
    outputs = Activation(activation=args.activation, name="activation3")(outputs)

    if args.regression_mode in (0, 1):
        outputs = tf.keras.layers.Dense(
            units=len(args.selected_items),
            activation="sigmoid" if args.regression_mode == 0 else None,
            name="outputs",
        )(outputs)
    else:
        raise NotImplementedError(
            f"regression mode {args.regression_mode} has not been implemented."
        )

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
