import tensorflow as tf


def VAE(input_dim,
        timesteps,
        batch_size,
        intermediate_dim,
        latent_dim,
        epsilon_std=1.):
    x = tf.keras.layers.Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = tf.keras.layers.LSTM(intermediate_dim)(x)

    # VAE Z layer
    z_mean = tf.keras.layers.Dense(latent_dim)(h)
    z_log_sigma = tf.keras.layers.Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = tf.keras.backend.random_normal(
            shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # latent variable
    z = tf.keras.layers.Lambda(sampling, output_shape=(
        latent_dim,))([z_mean, z_log_sigma])

    # decoded LSTM layer
    decoder_h = tf.keras.layers.LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = tf.keras.layers.LSTM(input_dim, return_sequences=True)

    h_decoded = tf.keras.layers.RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = tf.keras.models.Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = tf.keras.models.Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = tf.keras.layers.Input(shape=(latent_dim,))

    _h_decoded = tf.keras.layers.RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = tf.keras.models.Model(decoder_input, _x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = tf.keras.losses.mean_squared_error(x, x_decoded_mean)
        kl_loss = - 0.5 * tf.keras.backend.mean(1 + z_log_sigma -
                                                tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return vae, encoder, generator
