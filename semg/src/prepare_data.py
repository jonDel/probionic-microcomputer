"""Prepare sEMG data to be used to train a machine learning model."""
from types import FunctionType
from typing import List
from bisect import bisect_left
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


def sync_classes(measures_file: str, classes_file: str,
                 subsamp_rate: int = 1) -> DataFrame:
    """Fill the labels between first and last labels of an interval.

    Args:
        measures_file (str): location of the file with the sEMG measures.
        classes_file (str): location of the file with the labels of the
            begining and the end of each interval.
        subsamp_rate (int, optional): subsampling rate. Defaults to 1.

    Returns:
        DataFrame: a dataframe with a proper label for each sEMG point.
    """
    m_df = pd.read_csv(measures_file)[::subsamp_rate]
    c_df = pd.read_csv(classes_file)
    m_df["class"] = [0]*len(m_df)
    for index, row in m_df.iterrows():
        idx = bisect_left(c_df['time'].values, row['time'])
        if idx < len(c_df):
            m_class = c_df["class"].iloc[idx-1]
            m_df.at[index, "class"] = m_class
    return m_df


def combine_datasets(dataframes: List[DataFrame]) -> DataFrame:
    """Combine the datasets of the different movements into one dataframe.

    Args:
        dataframes (List[DataFrame]): a list of dataframes of each different
            movement.

    Returns:
        DataFrame: a dataframe with the SEMG signals of all movements.
    """
    time_offset = 0
    cp_dataframes = [df.copy() for df in dataframes]
    for dataframe in cp_dataframes:
        dataframe["time"] = dataframe["time"] + time_offset
        time_offset = dataframe["time"].iloc[-1]
    return pd.concat(cp_dataframes)


def preprocess(dataframe: DataFrame, window_size: float,
               step_size: float, freq: int = 380) -> DataFrame:
    """Create sliding windows of window_size and step_size seconds advance.

    Args:
        dataframe (DataFrame): input dataframe
        window_size (float): sliding window size, in seconds.
        step_size (float): sliding window advance
        freq (int, optional): number of SEMG data points per second.
            Defaults to 380.

    Returns:
        DataFrame: dataframe with data organized in sliding windows.
    """
    scaler = StandardScaler()
    dataframe[['semg']] = scaler.fit_transform(dataframe[['semg']])
    n_window = int(freq*window_size)
    n_step = int(freq*step_size)
    m_class = []
    m_myo = []
    n_total = int(len(dataframe)/n_step - int(n_window/n_step))
    for stp in range(n_total):
        window = dataframe["semg"].iloc[stp*n_step:stp*n_step + n_window]
        m_myo.append(window.to_list())
        m_class.append(dataframe["class"].iloc[stp*n_step + n_window])
    return pd.DataFrame(list(zip(m_myo, m_class)),
                        columns=["semg", "class"])


def root_mean_square(x: ndarray) -> float:
    """Perform the root mean square on an array.

    Args:
        x (ndarray): numpy array

    Returns:
        float: the root mean square value of the array
    """
    return np.sqrt(np.mean(np.square(x), axis=0))


def get_train_data(
        dataframe: DataFrame,
        backend: str = "keras",
        overlap_step: float = 0.01,
        time_step_window: float = 0.2,
        test_size: float = 0.3,
        agg_func: FunctionType = root_mean_square,
        n_channels: int = 1) -> List[ndarray]:
    """Return data ready to train a machine learning model.

    Args:
        dataframe (DataFrame): dataframe with SEMG data.
        backend (str, optional): framework that will be used later to train
            the model. Could be keras or sklearn. Defaults to "keras".
        overlap_step (float, optional): sliding window advance, in seconds.
             Defaults to 0.01.
        time_step_window (float, optional): sliding window size, in seconds.
             Defaults to 0.2.
        test_size (float, optional): proportion of data to use as test after
            the training. Defaults to 0.3.
        agg_func (FunctionType, optional): function to be applied to each time
            window, when using sklearn as backend. Defaults to root mean
            square.
        n_channels (int, optional): number of SEMG channels. Defaults to 1.

    Returns:
        List[ndarray]: data divided into train (input and output) and test
            (input and output) datasets, ready to be fed to a machine learning
            algorithm.
    """
    freq = len(dataframe)/(dataframe["time"].iloc[-1]
                           - dataframe["time"].iloc[0])
    prep_df = preprocess(dataframe, time_step_window, overlap_step, freq)
    if backend == "sklearn":
        X = prep_df["semg"].apply(agg_func)
        y = prep_df["class"]
    elif backend == "keras":
        np_myo = np.array(prep_df["semg"].to_list())
        X = np_myo.reshape(np_myo.shape[0], len(np_myo[0]), n_channels, 1)
        y = tf.keras.utils.to_categorical(prep_df["class"].to_numpy(),
                                          max(prep_df["class"] + 1))
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                                        test_size=test_size)
    return X_train, X_test, Y_train, Y_test
