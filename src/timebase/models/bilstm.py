from .registry import register

import tensorflow as tf

from timebase.models import embedding


@register("bilstm")
def get_model(args, name: str = "BiLSTM"):
    inputs, embeddings = embedding.encoder(args)

    outputs = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(
            units=args.num_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            dropout=args.dropout,
            recurrent_dropout=args.r_dropout,
            unroll=False,
        ),
        merge_mode="concat",
        name="BiLSTM",
    )(embeddings)

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
