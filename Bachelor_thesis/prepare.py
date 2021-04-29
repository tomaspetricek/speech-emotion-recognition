import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from config import DATASET_PATH
from data import (Dataset, RAVDESSLabel, TESSLabel,
                  EMOVOLabel, SAVEELabel, MFCCData, WAVData,
                  RAVDESSUnifiedLabel, TESSUnifiedLabel, SAVEEUnifiedLabel,
                  EMOVOUnifiedLabel, CallCentersUnifiedLabel,
                  FOUR_EMOTIONS_CONVERSION_TABLE, THREE_EMOTIONS_CONVERSION_TABLE)

from files import DatasetInfoFile, SetInfoFile
from data import NEUTRAL, ANGER, FEAR, SADNESS, HAPPINESS, DISGUST, SURPRISE
from tools import NDScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preparer:

    TRAIN_DIR = "train"
    TEST_DIR = "test"
    VAL_DIR = "val"
    WHOLE_DIR = "whole"

    def __init__(self, datasets, test_size=None, val_size=None, conversion_table=None):
        self.datasets = datasets
        self.val_size = val_size
        self.test_size = test_size
        self.samples = self.labels = None
        self.conversion_table = conversion_table

    def load_data(self):
        """
        Load in datasets and returns X and y as numpy arrays.
        """
        self.samples = []
        self.labels = []

        for dataset in self.datasets:
            # get samples
            samples = dataset.samples
            labels = dataset.labels

            self.samples.append(samples)

            self.labels.append(np.array(labels).flatten())

    def convert_labels(self):
        converted_labels = []
        for labels_dataset in self.labels:
            converted_labels_dataset = []
            for label in labels_dataset:
                converted_label = self.conversion_table[label]
                converted_labels_dataset.append(converted_label)

            converted_labels.append(np.array(converted_labels_dataset))

        self.labels = converted_labels

    def split_data(self, frames, labels, test_size):
        """
        Splits th data into train. validation and test set.
        """
        random_state = 42

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            frames,
            labels,
            stratify=labels,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, y_train, X_test, y_test

    def save_set(self, samples, labels, root_dirname, set_dir):
        set_dirname = os.path.join(root_dirname, set_dir)
        os.mkdir(set_dirname)

        n_samples = len(samples)

        samples_lengths = []
        for index, sample in enumerate(samples):
            samples_lengths.append(sample.shape[0])

        # convert to numpy array
        frames = np.concatenate(samples, axis=0)
        labels = np.array(labels).flatten()

        sample_filename = "samples.npy"
        samples_path = os.path.join(set_dirname, sample_filename)
        np.save(samples_path, frames)

        label_filename = "labels.npy"
        labels_path = os.path.join(set_dirname, label_filename)
        np.save(labels_path, labels)

        info_path = os.path.join(set_dirname, "info.txt")
        SetInfoFile(path=info_path).write(n_samples, samples_lengths, sample_filename, label_filename)

    def __call__(self, result_dirname):
        self.load_data()

        # convert labels
        if self.conversion_table:
            self.convert_labels()

        dataset_index = 0
        # save dataset info
        n_classes = len(np.unique(self.labels[dataset_index]))
        n_features = self.samples[dataset_index][0].shape[1]

        n_samples = 0
        for samples in self.samples:
            n_samples += len(samples)

        info_path = os.path.join(result_dirname, "info.txt")
        DatasetInfoFile(path=info_path).write(n_features, n_classes, n_samples)

        X_train = []
        y_train = np.array([], dtype=np.int32)
        X_test = []
        y_test = np.array([], dtype=np.int32)
        X_val = []
        y_val = np.array([], dtype=np.int32)

        for dataset_index in range(len(self.datasets)):
            samples = self.samples[dataset_index]
            labels = self.labels[dataset_index]

            X_train_local, y_train_local, X_test_full, y_test_full = self.split_data(samples, labels, self.test_size)

            X_test_local, y_test_local, X_val_local, y_val_local = self.split_data(X_test_full, y_test_full, self.val_size)

            # add sets
            X_train = X_train + X_train_local
            y_train = np.concatenate((y_train, y_train_local))

            X_test = X_test + X_test_local
            y_test = np.concatenate((y_test, y_test_local))

            X_val = X_val + X_val_local
            y_val = np.concatenate((y_val, y_val_local))

        # save test set
        self.save_set(X_test, y_test, result_dirname, self.TEST_DIR)

        # save train set
        self.save_set(X_train, y_train, result_dirname, self.TRAIN_DIR)

        # save val set
        self.save_set(X_val, y_val, result_dirname, self.VAL_DIR)

def main():
    # load ravdess
    ravdess_path = DATASET_PATH.format(language="english", name="RAVDESS", form="mfcc")
    ravdess_mfcc_unified = Dataset(ravdess_path, MFCCData(), RAVDESSUnifiedLabel())
    #
    # load tess
    tess_path = DATASET_PATH.format(language="english", name="TESS", form="mfcc")
    tess_mfcc_unified = Dataset(tess_path, MFCCData(), TESSUnifiedLabel())

    # load savee
    savee_path = DATASET_PATH.format(language="english", name="SAVEE", form="mfcc")
    savee_mfcc_unified = Dataset(savee_path, MFCCData(), SAVEEUnifiedLabel())

    # load emovo
    emovo_path = DATASET_PATH.format(language="italian", name="EMOVO", form="mfcc")
    emovo_mfcc_unified = Dataset(emovo_path, MFCCData(), EMOVOUnifiedLabel())

    datasets = [
        ravdess_mfcc_unified,
        tess_mfcc_unified,
        savee_mfcc_unified,
        emovo_mfcc_unified
    ]

    preperer = Preparer(
        datasets=datasets,
        test_size=0.2,
        val_size=0.5,
        conversion_table=THREE_EMOTIONS_CONVERSION_TABLE,
    )

    result_dir = "prepared_data/int-3-re-80-10-10"
    os.mkdir(result_dir)
    preperer(result_dir)


if __name__ == "__main__":
    main()
