# from processing.convertors import AudioFormatConverter, MFCCConverter
# from classes import Dataset
# from pprint import pprint
import numpy as np
# from enums import DATASET_PATH
# from files import TextFile, HTKFile
# from os_utils import change_file_extension
# import re

def get_n_samples(path):
    dataset = np.load(path)
    return dataset.shape[0]


if __name__ == "__main__":
    train_path = "prepared_data/fullset_npy_2/train/samples_0.npy"
    test_path = "prepared_data/fullset_npy_2/test/samples_0.npy"
    val_path = "prepared_data/fullset_npy_2/val/samples_0.npy"

    n_test_samples = get_n_samples(test_path)
    print(n_test_samples)
    n_train_samples = get_n_samples(train_path)
    print(n_train_samples)
    n_val_samples = get_n_samples(val_path)
    print(n_val_samples)
    total_n_samples = n_test_samples + n_train_samples + n_val_samples

    print(total_n_samples)









