import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import DATASET_PATH
from datasets import (Dataset, RAVDESSLabel, TESSLabel,
                      EMOVOLabel, SAVEELabel, MFCCData, WAVData,
                      RAVDESSUnifiedLabel, TESSUnifiedLabel, SAVEEUnifiedLabel,
                      EMOVOUnifiedLabel)

from tools import add_margin, IndexPicker


def load_data():
    """
    Load in datasets and returns X and y as numpy arrays.
    """
    X_column = "coefficients"
    y_column = "emotion"

    # load ravdess
    ravdess_path = DATASET_PATH.format(language="english", name="RAVDESS", form="mfcc")
    ravdess_mfcc_unified = Dataset(ravdess_path, MFCCData(), RAVDESSUnifiedLabel())

    # load tess
    tess_path = DATASET_PATH.format(language="english", name="TESS", form="mfcc")
    tess_mfcc_unified = Dataset(tess_path, MFCCData(), TESSUnifiedLabel())

    # load savee
    savee_path = DATASET_PATH.format(language="english", name="SAVEE", form="mfcc")
    savee_mfcc_unified = Dataset(savee_path, MFCCData(), SAVEEUnifiedLabel())

    # load emovo
    emovo_path = DATASET_PATH.format(language="italian", name="EMOVO", form="mfcc")
    emovo_mfcc_unified = Dataset(emovo_path, MFCCData(), EMOVOUnifiedLabel())

    # combine datasets
    ravdess_mfcc_unified.combine(savee_mfcc_unified, tess_mfcc_unified, emovo_mfcc_unified)
    dataset = ravdess_mfcc_unified

    # get samples
    samples = dataset.samples

    # convert to numpy array
    X = np.array(list(samples[X_column]))

    y = np.array(list(samples[y_column]))

    return X, y


def prepare_data(X, index_picker):
    """
    Adds margin and reshapes it so that each row represents one sample.
    """
    # add margin
    X_margined = np.array(add_margin(X, index_picker))

    # reshape
    n_samples, window_length, n_features = X_margined.shape
    X_reshaped = np.array(np.reshape(X_margined, (n_samples, -1)))

    return X_reshaped


def split_data(X, y):
    """
    Splits th data into train. validation and test set.
    """
    # split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.05,
        random_state=42
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        stratify=y_train_full,
        test_size=0.05,
        random_state=42
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test

