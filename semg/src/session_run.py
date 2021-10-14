"""Trains a deepconvlstm model on SEMG data."""
from typing import Tuple
import time
import re
import sys
import logging
from glob import glob
import pickle
from pathlib import Path
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, \
    EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
from model import deep_conv_lstm
from prepare_data import get_train_data, sync_classes, combine_datasets


K.set_image_data_format("channels_last")
K.set_learning_phase(1)
np.random.seed(1)
logger = logging.getLogger("deepconvlstm")
TRAIN_PARAMS = {
    "datasets_path": "../datasets",
    "weights_path": "../results/weights/",
    "log_dir": "../results/logs/",
    "history_path": "../results/history/",
    "window_factor": 0.1,
    "subsamp_rate": 1,
    "train_split": 3
}
WEIGHTS_PATTERN = "epoch:{epoch:02d}-accuracy:{accuracy:." + \
    "4f}-val_accuracy:{val_accuracy:." + \
    "4f}.hdf5"


def best_weight(folder: str, metric: str,
                file_pattern: str = "weights--{}--*.hdf5") -> Tuple[str, int]:
    """Return best weight based on metric.

    The weights file is supposed to be written similarly to
    weights--epoch:11-accuracy:0.5971-val_accuracy:0.4687.hdf5

    Args:
        folder (str): path of weight's folder
        metric (str): metric to use
            as comparision
        file_pattern (str, optional): file string pattern.
            Defaults to "weights--{}--*.hdf5".

    Returns:
        Tuple[str, int]: [description]
    """
    best_metric = 0
    weights_list = glob(folder + "/*")
    if not weights_list:
        return ("", 0)
    b_weight = weights_list[0]
    reg = re.compile("-" + metric + ":(\d.\d{4})")
    try:
        for filename in weights_list:
            res = reg.search(filename)
            if res:
                file_metric = (float(res.groups()[0]))
                if file_metric > best_metric:
                    best_metric = file_metric
                    b_weight = filename
        epoch = int(re.search("epoch:(\d+)-", b_weight).groups()[0])
    except AttributeError:
        return ("", 0)
    return (b_weight, epoch)


def load_pretrained(model: Sequential, weights_path: str,
                    metric: str = "accuracy") -> Tuple[Sequential, int]:
    """Return a model with the best pretrained weights from a folder.

    Args:
        model (Sequential):  model with basic structure
        weights_path (str): path to the model weights
        metric (str, optional): metric to use as comparision.
            Defaults to "accuracy".

    Raises:
        OSError: raises when no weight file is found.

    Returns:
        Tuple[Sequential, int]: a keras model loaded with best weights file,
            or None if no file could be found, and the corresponding epoch.
    """
    weight_file, epoch = best_weight(weights_path, metric)
    if not weight_file:
        raise OSError("Weight file not found!")
    model.load_weights(weight_file)
    return (model, epoch)


def get_model(n_time_steps: int, n_channels: int, n_classes: int,
              **kwargs) -> Sequential:
    """Return a deepconvlstm model with the best pretrained weights found.

    Args:
        n_time_steps (int): number of time steps in the recurrent layers
        n_channels (int): number of SEMG channels
        n_classes (int): number of distinct classes or movements

    Returns:
        Sequential: A keras deepconvlstm model
    """
    def_args = {
        "verbose": 1,
        "early_patience": 20,
        "plateau_patience": 5,
        "profile_batch": 2,
        "save_best_only": True,
        "monitor": "val_accuracy",
        "checkpoint_mode": "max",
        "min_lr": 0.001,
        "lr_factor": 0.2,
        "weights_path": "../results/weights/",
        "log_dir": "results/logs/",
        "history_path": "../results/history/",
        "weights_pattern": WEIGHTS_PATTERN,
        "write_images": True          
    }
    def_args.update(kwargs)
    model = deep_conv_lstm(n_time_steps, n_channels, n_classes, **kwargs)
    file_weights = def_args["weights_path"] + "/weights--" + \
        def_args["weights_pattern"]
    checkpoint = ModelCheckpoint(file_weights, verbose=def_args["verbose"],
                                 monitor=def_args["monitor"],
                                 save_best_only=def_args["save_best_only"],
                                 mode=def_args["checkpoint_mode"])
    tensorboard = TensorBoard(log_dir=def_args["log_dir"]+"{}".format(
                              time.strftime("%d/%m/%Y--%H:%M:%S")),
                              profile_batch=def_args["profile_batch"],
                              write_images=def_args["write_images"])
    early_stopping = EarlyStopping(monitor=def_args["monitor"],
                                   patience=def_args["early_patience"],
                                   verbose=def_args["verbose"])
    reduce_lr = ReduceLROnPlateau(monitor=def_args["monitor"],
                                  factor=def_args["lr_factor"],
                                  patience=def_args["plateau_patience"],
                                  min_lr=def_args["min_lr"])
    callbacks_list = [checkpoint, tensorboard, early_stopping, reduce_lr]
    return (model, callbacks_list, def_args)


def run_training(
        dataframe: DataFrame,
        epochs: int = 150,
        overlap_step: float = 0.01,
        time_step_window: float = 0.2,
        train_params: dict = TRAIN_PARAMS,
        **kwargs) -> Tuple[ndarray, ndarray, ndarray, Sequential]:
    """Trains a deepconvlstm model on SEMG data.

    Args:
        dataframe (DataFrame): dataframe with annotated SEMG data
        epochs (int, optional): number of training epochs. Defaults to 150.
        overlap_step (float, optional): sliding window advance, in seconds.
             Defaults to 0.01.
        time_step_window (float, optional): sliding window size, in seconds.
             Defaults to 0.2.
        train_params (dict, optional): specific training parameters.
             Defaults to TRAIN_PARAMS.

    Returns:
        Tuple[ndarray, ndarray, ndarray, Sequential]: a tuple of a predictions
            array, a true labels array, a predictions probability array and the
            trained deepconvlstm model
    """
    def_args = {
        "filters": [64, 64, 64, 64],
        "lstm_dims": [128, 64],
        "learn_rate": 0.001,
        "decay_factor": 0.9,
        "reg_rate": 0.01,
        "metrics": ["accuracy"],
        "weight_init": "lecun_uniform",
        "dropout_prob": 0.5,
        "lstm_activation": "tanh",
        "validation_split": 0.1,
        "batch_size": None
    }
    def_args.update(kwargs)
    w_folder = train_params["weights_path"]
    Path(w_folder).mkdir(parents=True, exist_ok=True)
    Path(train_params["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(train_params["history_path"]).mkdir(parents=True, exist_ok=True)
    X_train, X_test, Y_train, Y_test = get_train_data(
        dataframe, "keras", overlap_step, time_step_window)
    y_true = [np.argmax(el) for el in Y_test]
    n_time_steps = X_train.shape[1]
    n_channels = X_train.shape[2]
    n_classes = max(y_true) + 1
    model, callbacks_list, params = get_model(n_time_steps, n_channels,
                                              n_classes, **kwargs)
    res = load_pretrained(model, params["weights_path"])
    if res[0]:
        initial_epoch = res[1]
        logger.debug("Using pre-trained weights... resuming from epoch {}".
                     format(initial_epoch))
        model = res[0]
    else:
        initial_epoch = 0
    model.summary()
    init = time.time()
    hist = model.fit(X_train, Y_train, epochs=epochs,
                     batch_size=def_args["batch_size"],
                     validation_split=def_args["validation_split"],
                     callbacks=callbacks_list,
                     verbose=params["verbose"],
                     initial_epoch=initial_epoch)
    training_time = time.time() - init
    logger.info(f"Model trained in {training_time} seconds.")
    wfile, epoch = best_weight(w_folder, "val_accuracy")
    logger.debug("Best results from epoch {}, saved in file {}".
                 format(epoch, wfile))
    logger.debug("Saving history in a picke file...")
    filehistname = train_params["history_path"] + "history.pickle"
    with open(filehistname, "wb") as fname:
        pickle.dump(hist.history, fname)
    test_accuracy = model.evaluate(X_test, Y_test)[1]
    train_accuracy = model.evaluate(X_train, Y_train)[1]
    predictions_prob = model.predict(X_test)
    predictions = [np.argmax(pred) for pred in predictions_prob]
    logger.info("Train Accuracy = " + str(train_accuracy))
    logger.info("Test Accuracy = " + str(test_accuracy))
    return predictions, y_true, predictions_prob, model


def get_combined_df(datasets_folder: str = "../datasets") -> DataFrame:
    """Combine the SEMG dataframes of each class into one dataframe.

    Args:
        datasets_folder (str, optional): the datasets folder path.
            Defaults to "../datasets".

    Returns:
        DataFrame: a dataframe with all 4 movements SEMG data combined.
    """
    df_hold_ball = sync_classes(
        datasets_folder + "/raw/power_sphere.csv",
        datasets_folder + "/annotated_classes/power_sphere_classes.csv")
    df_hold_tripod_ball = sync_classes(
        datasets_folder + "/raw/tripod.csv",
        datasets_folder + "/annotated_classes/tripod_classes.csv")
    df_hold_cup = sync_classes(
        datasets_folder + "/raw/medium_wrap.csv",
        datasets_folder + "/annotated_classes/medium_wrap_classes.csv")
    df_hold_card = sync_classes(
        datasets_folder + "/raw/lateral.csv",
        datasets_folder + "/annotated_classes/lateral_classes.csv")
    return combine_datasets(
        [
            df_hold_cup,
            df_hold_card,
            df_hold_tripod_ball,
            df_hold_ball
        ]
    )


if __name__ == "__main__":
    hdlr = logging.FileHandler("deepconvlstm.log")
    stdout_hdlr = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    stdout_hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.addHandler(stdout_hdlr)
    logger.setLevel(logging.DEBUG)
    predictions, y_true, predictions_prob, model = \
        run_training(get_combined_df())
