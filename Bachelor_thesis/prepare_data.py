import os
import numpy as np

from sklearn.model_selection import train_test_split

from config import DATASET_PATH
from datasets import (Dataset, RAVDESSLabel, TESSLabel,
                      EMOVOLabel, SAVEELabel, MFCCData, WAVData,
                      RAVDESSUnifiedLabel, TESSUnifiedLabel, SAVEEUnifiedLabel,
                      EMOVOUnifiedLabel)

from files import DatasetInfoFile, SetInfoFile


class Preparer:

    def __init__(self, datasets, test_size, val_size):
        self.datasets = datasets
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
        labels = dataset.labels

        # convert to numpy array
        self.X = samples

        self.y = labels

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

    def save_set(self, X, y, directory):
        os.mkdir(directory)

        n_samples = len(X)

        samples_lengths = []
        for index, samples in enumerate(X):
            samples_lengths.append(samples.shape[0])

        samples = np.concatenate(X, axis=0)
        sample_filename = "samples.npy"
        samples_path = os.path.join(directory, sample_filename)
        np.save(samples_path, samples)

        labels = np.array(y)
        label_filename = "labels.npy"
        labels_path = os.path.join(directory, label_filename)
        np.save(labels_path, labels)

        info_path = os.path.join(directory, "info.txt")
        SetInfoFile(path=info_path).write(n_samples, samples_lengths, sample_filename, label_filename)

    def __call__(self, result_dir):
        self.load_data()

        # save dataset info
        n_classes = len(np.unique(self.y))
        n_features = self.X[0].shape[1]
        n_samples = len(self.X)

        info_path = os.path.join(result_dir, "info.txt")
        DatasetInfoFile(path=info_path).write(n_features, n_classes, n_samples)

        X_train, y_train, X_valid, y_valid, X_test, y_test = self.split_data()

        train_dir = os.path.join(result_dir, "train")
        self.save_set(X_train, y_train, train_dir)

        val_dir = os.path.join(result_dir, "val")
        self.save_set(X_valid, y_valid, val_dir)

        test_dir = os.path.join(result_dir, "test")
        self.save_set(X_test, y_test, test_dir)


if __name__ == "__main__":
    preperer = Preparer(
        datasets=None,
        test_size=0.05,
        val_size=0.05,
    )

    # chunk_size =10**8
    result_dir = "prepared_data/fullset_npy_3"
    os.mkdir(result_dir)
    preperer(result_dir)

