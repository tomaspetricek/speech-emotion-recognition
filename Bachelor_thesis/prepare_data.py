import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split

from config import DATASET_PATH
from datasets import (Dataset, RAVDESSLabel, TESSLabel,
                      EMOVOLabel, SAVEELabel, MFCCData, WAVData,
                      RAVDESSUnifiedLabel, TESSUnifiedLabel, SAVEEUnifiedLabel,
                      EMOVOUnifiedLabel, CallCentersUnifiedLabel,
                      FOUR_EMOTIONS_CONVERSION_TABLE)

from files import DatasetInfoFile, SetInfoFile
from datasets import NEUTRAL, ANGER, FEAR, SADNESS, HAPPINESS, DISGUST, SURPRISE
from tools import NDScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preparer:

    TRAIN_DIR = "train"
    TEST_DIR = "test"
    VAL_DIR = "val"
    WHOLE_DIR = "whole"

    def __init__(self, dataset, test_size=None, val_size=None, conversion_table=None, scaler=None):
        self.dataset = dataset
        self.val_size = val_size
        self.test_size = test_size
        self.samples = self.labels = None
        self.conversion_table = conversion_table
        self.scaler = scaler

    def load_data(self):
        """
        Load in datasets and returns X and y as numpy arrays.
        """

        # get samples
        samples = self.dataset.samples
        labels = self.dataset.labels

        # convert to numpy array
        self.samples = samples

        self.labels = np.array(labels).flatten()

    def convert_labels(self):
        converted_labels = []
        for label in self.labels:
            converted_label = self.conversion_table[label]
            converted_labels.append(converted_label)

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
        labels = np.array(labels)

        # scale samples
        if self.scaler:
            if set_dir == self.TRAIN_DIR:
                frames = self.scaler.fit_transform(frames)
            else:
                frames = self.scaler.transform(frames)

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

        # save dataset info
        n_classes = len(np.unique(self.labels))
        n_features = self.samples[0].shape[1]
        n_samples = len(self.samples)

        info_path = os.path.join(result_dirname, "info.txt")
        DatasetInfoFile(path=info_path).write(n_features, n_classes, n_samples)

        if self.test_size:
            X_train_full, y_train_full, X_test, y_test = self.split_data(self.samples, self.labels, self.test_size)

            if self.val_size:
                X_train, y_train, X_val, y_val = self.split_data(X_train_full, y_train_full, self.val_size)

                # save train set
                self.save_set(X_train, y_train, result_dirname, self.TRAIN_DIR)

                # save val set
                self.save_set(X_val, y_val, result_dirname, self.VAL_DIR)
            else:
                # save train set
                self.save_set(X_train_full, y_train_full, result_dirname, self.TRAIN_DIR)

            # save test set
            self.save_set(X_test, y_test, result_dirname, self.TEST_DIR)
        else:
            self.save_set(self.samples, self.labels, result_dirname, self.WHOLE_DIR)

def main():
    # # load ravdess
    # ravdess_path = DATASET_PATH.format(language="english", name="RAVDESS", form="mfcc")
    # ravdess_mfcc_unified = Dataset(ravdess_path, MFCCData(), RAVDESSUnifiedLabel())
    #
    # # load tess
    # tess_path = DATASET_PATH.format(language="english", name="TESS", form="mfcc")
    # tess_mfcc_unified = Dataset(tess_path, MFCCData(), TESSUnifiedLabel())
    #
    # # load savee
    # savee_path = DATASET_PATH.format(language="english", name="SAVEE", form="mfcc")
    # savee_mfcc_unified = Dataset(savee_path, MFCCData(), SAVEEUnifiedLabel())

    # load emovo
    emovo_path = DATASET_PATH.format(language="italian", name="EMOVO", form="mfcc")
    emovo_mfcc_unified = Dataset(emovo_path, MFCCData(), EMOVOUnifiedLabel())
    dataset = emovo_mfcc_unified

    # # combine datasets
    # ravdess_mfcc_unified.combine(savee_mfcc_unified, tess_mfcc_unified)
    # dataset = ravdess_mfcc_unified

    # call_center_path = DATASET_PATH.format(language="czech", name="CallCenters", form="mfcc")
    # call_center_unified = Dataset(call_center_path, MFCCData(), CallCentersUnifiedLabel())
    # dataset = call_center_unified

    # standard_scaler = NDScaler(StandardScaler())

    with open("prepared_data/en-4-stdsc-90-10/scaler.obj", 'rb') as file:
        en_4_scaler = pickle.load(file)

    preperer = Preparer(
        dataset=dataset,
        # test_size=0.1,
        conversion_table=FOUR_EMOTIONS_CONVERSION_TABLE,
        scaler=en_4_scaler,
    )

    result_dir = "prepared_data/cz-4-stdsc"
    os.mkdir(result_dir)
    preperer(result_dir)

    # scaler_fitted = preperer.scaler
    # scaler_filename = os.path.join(result_dir, "scaler.obj")
    #
    # with open(scaler_filename, "wb") as file:
    #     pickle.dump(scaler_fitted, file)


if __name__ == "__main__":
    main()
