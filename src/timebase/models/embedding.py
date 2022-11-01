import tensorflow as tf


def encoder(args):
    """Encoding layer to process input channels"""
    channel_names = sorted(args.input_shapes.keys())
    channel_inputs = {
        channel: tf.keras.Input(args.input_shapes[channel], name=channel)
        for channel in channel_names
    }
    expand_dims = tf.keras.layers.Reshape((-1, 1), name="expand_dims")
    # learn embeddings if segment is not time aligned
    if args.time_alignment == 0:
        embeddings = []
        if args.embedding_type == 0:
            # create GRU layers for each channel
            for channel in channel_names:
                embedding = tf.keras.layers.GRU(
                    args.embedding_dim, name=f"emb_{channel}"
                )(expand_dims(channel_inputs[channel]))
                embedding = expand_dims(embedding)
                embeddings.append(embedding)
        elif args.embedding_type == 1:
            # create MLP layer for each channel
            for channel in channel_names:
                embedding = tf.keras.layers.Dense(
                    args.embedding_dim, activation="gelu", name=f"emb_{channel}"
                )(channel_inputs[channel])
                embedding = expand_dims(embedding)
                embeddings.append(embedding)
        else:
            raise NotImplementedError(
                f"padding_mode {args.embedding_type} has not been implemented."
            )
    else:
        embeddings = [expand_dims(channel_inputs[channel]) for channel in channel_names]

    embeddings = tf.keras.layers.Concatenate(axis=-1, name="embeddings")(embeddings)

    return channel_inputs, embeddings
