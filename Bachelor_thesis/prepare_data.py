import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import DATASET_PATH
from datasets import (Dataset, RAVDESSLabel, TESSLabel,
                      EMOVOLabel, SAVEELabel, MFCCData, WAVData,
                      RAVDESSUnifiedLabel, TESSUnifiedLabel, SAVEEUnifiedLabel,
                      EMOVOUnifiedLabel)

from tools import add_margin, IndexPicker
from files import TextFile


class Preparer:

    def __init__(self, datasets, index_picker, test_size, val_size):
        self.datasets = datasets
        self.index_picker = index_picker
        self.val_size = val_size
        self.test_size = test_size
        self.X = self. y = None

    def load_data(self):    # TODO Make it universal
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
        self.X = np.array(list(samples[X_column]))

        self.y = np.array(list(samples[y_column]))

    def transform_data(self):
        """
        Adds margin and reshapes it so that each row represents one sample.
        """
        # add margin
        self.X = np.array(add_margin(self.X, self.index_picker))

        # reshape
        n_samples, window_length, n_features = self.X.shape
        self.X = np.array(np.reshape(self.X, (n_samples, -1)))

    def split_data(self):
        """
        Splits th data into train. validation and test set.
        """
        random_state = 42

        # split data
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            self.X,
            self.y,
            stratify=self.y,
            test_size=self.test_size,
            random_state=random_state
        )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full,
            y_train_full,
            stratify=y_train_full,
            test_size=self.val_size,
            random_state=random_state
        )

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    @staticmethod
    def save_data(X, y, dir):
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        samples_path = os.path.join(dir, "samples.csv")
        X.to_csv(samples_path, index=False)

        labels_path = os.path.join(dir, "labels.csv")
        y.to_csv(labels_path, index=False)

    @staticmethod
    def save_data_numpy(X, y, dir):
        samples_path = os.path.join(dir, "samples.npy")
        np.save(samples_path, X)

        labels_path = os.path.join(dir, "labels.npy")
        np.save(labels_path, y)

    @staticmethod
    def save_data_numpy_split(X, y, directory, chunk_size=10**8):
        os.mkdir(directory)

        n_chunks = round(X.shape[0] * X.shape[1] / chunk_size)
        chunk_sizes = []
        samples_filenames = []
        labels_filenames = []

        if n_chunks == 0:
            n_chunks = 1

        for index, samples in enumerate(np.array_split(X, n_chunks)):
            filename = "samples_{}.npy".format(index)
            samples_path = os.path.join(directory, filename)
            np.save(samples_path, samples)

            chunk_sizes.append(samples.shape[0])
            samples_filenames.append(filename)

        for index, labels in enumerate(np.array_split(y, n_chunks)):
            filename = "labels_{}.npy".format(index)
            labels_path = os.path.join(directory, filename)
            np.save(labels_path, labels)

            labels_filenames.append(filename)

        info_path = os.path.join(directory, "info.txt")
        info = [n_chunks] + chunk_sizes
        info_str = list(map(str, info)) + samples_filenames + labels_filenames
        TextFile(path=info_path).write_lines(info_str)

    def __call__(self, result_dir):
        self.load_data()

        self.transform_data()

        n_classes = len(np.unique(self.y))
        n_features = self.X.shape[1]
        n_samples = self.X.shape[0]

        # save dataset info
        info_path = os.path.join(result_dir, "info.txt")
        info = [
            n_features,
            n_classes,
            n_samples,
        ]

        info_str = list(map(str, info))
        TextFile(path=info_path).write_lines(info_str)

        X_train, y_train, X_valid, y_valid, X_test, y_test = self.split_data()

        train_dir = os.path.join(result_dir, "train")
        self.save_data_numpy_split(X_train, y_train, train_dir)

        val_dir = os.path.join(result_dir, "val")
        self.save_data_numpy_split(X_valid, y_valid, val_dir)

        test_dir = os.path.join(result_dir, "test")
        self.save_data_numpy_split(X_test, y_test, test_dir)


if __name__ == "__main__":
    preperer = Preparer(
        datasets=None,
        index_picker=IndexPicker(25, 25),
        test_size=0.05,
        val_size=0.05
    )

    result_dir = "prepared_data/fullset_npy_split"
    os.mkdir(result_dir)
    preperer(result_dir)

