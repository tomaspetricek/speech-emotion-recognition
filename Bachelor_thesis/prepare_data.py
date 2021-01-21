import numpy as np
import torch
import os

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
        annotations = []
        for index in range(len(y)):
            sample = torch.from_numpy(X[index])
            label = y[index]

            basename = "{}_{}.pt".format(index + 1, label)
            path = os.path.join(dir, basename)
            torch.save(sample, path)

        annotations_path = os.path.join(dir, "annotations.txt")
        TextFile(path=annotations_path).write_lines(annotations)

    def __call__(self, result_dir):
        self.load_data()

        self.transform_data()

        X_train, y_train, X_valid, y_valid, X_test, y_test = self.split_data()

        train_dir = os.path.join(result_dir, "train")
        os.mkdir(train_dir)
        self.save_data(X_train, y_train, train_dir)

        val_dir = os.path.join(result_dir, "val")
        os.mkdir(val_dir)
        self.save_data(X_valid, y_valid, val_dir)

        test_dir = os.path.join(result_dir, "test")
        os.mkdir(test_dir)
        self.save_data(X_test, y_test, test_dir)


if __name__ == "__main__":
    preperer = Preparer(
        datasets=None,
        index_picker=IndexPicker(1, 1),
        test_size=0.05,
        val_size=0.05
    )

    result_dir = "prepared_data/fullset"
    os.mkdir(result_dir)
    preperer(result_dir)

