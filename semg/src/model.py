"""Define a deep convolutional LSTM model."""
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, \
    Lambda, LSTM, Dropout, Reshape, TimeDistributed, Convolution2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import tensorflow as tf


tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
K.set_image_data_format("channels_last")
K.set_learning_phase(1)
HAS_GPU = bool(tf.test.gpu_device_name())


def deep_conv_lstm(n_time_steps: int, n_channels: int,
                   n_classes: int, **kwargs) -> Sequential:
    """Generate a model with convolution and LSTM layers.

    See Ordonez et al., 2016, http://dx.doi.org/10.3390/s16010115

    Args:
        n_time_steps (int): number of time steps in the recurrent layers
        n_channels (int): number of SEMG channels
        n_classes (int): number of distinct classes or movements

    Returns:
        Sequential: A keras deepconvlstm model
    """
    def_args = {
        'filters': [64, 64, 64, 64],
        'lstm_dims': [128, 64],
        'learn_rate': 0.001,
        'decay_factor': 0.9,
        'reg_rate': 0.01,
        'metrics': ['accuracy'],
        'weight_init': 'lecun_uniform',
        'dropout_prob': 0.5,
        'lstm_activation': 'tanh'
    }
    np.random.seed(1)
    def_args.update(kwargs)
    output_dim = n_classes  # number of classes
    weight_init = def_args['weight_init']  # weight initialization
    model = Sequential()  # initialize model
    model.add(BatchNormalization(input_shape=(n_time_steps, n_channels, 1)))
    for filt in def_args['filters']:
        # filt: number of filters used in a layer
        # filters: vector of filt values
        model.add(
            Convolution2D(filt, kernel_size=(3, 1), padding='same',
                          kernel_regularizer=l2(def_args['reg_rate']),
                          kernel_initializer=weight_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    # reshape 3 dimensional array back into a 2 dimensional array,
    # but now with more dept as we have the the filters for each channel
    model.add(Reshape(target_shape=(n_time_steps,
                                    def_args['filters'][-1] * n_channels)))
    for lstm_dim in def_args['lstm_dims']:
        # dropout before the dense layer
        model.add(Dropout(def_args['dropout_prob']))
        if HAS_GPU:
            model.add(LSTM(units=lstm_dim, return_sequences=True))
        else:
            model.add(LSTM(units=lstm_dim, return_sequences=True,
                           activation=def_args['lstm_activation']))
    # set up final dense layer such that every timestamp is given one
    # classification
    model.add(
        TimeDistributed(
            Dense(units=output_dim,
                  kernel_regularizer=l2(def_args['reg_rate']))))
    model.add(Activation("softmax"))
    # Final classification layer - per timestep
    model.add(Lambda(lambda x: x[:, -1, :], output_shape=[output_dim]))
    optimizer = optimizers.RMSprop(learning_rate=def_args['learn_rate'],
                                   rho=def_args['decay_factor'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=def_args['metrics'])
    return model
