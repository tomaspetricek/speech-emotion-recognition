import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import DATASET_PATH
from datasets import (Dataset, RAVDESSLabel, TESSLabel,
                      EMOVOLabel, SAVEELabel, MFCCData, WAVData,
                      RAVDESSUnifiedLabel, TESSUnifiedLabel, SAVEEUnifiedLabel,
                      EMOVOUnifiedLabel)

from tools import add_margin, IndexPicker
from classifiers import Sequential


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


def load_prepared_data(data_files):
    """
    Loads prepared data.
    """
    X_train = np.load(data_files['X_train'])

    y_train = np.load(data_files['y_train'])

    X_valid = np.load(data_files['X_valid'])

    y_valid = np.load(data_files['y_valid'])

    X_test = np.load(data_files['X_test'])

    y_test = np.load(data_files['y_test'])

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def prepare_torch_data_loader(X, y, batch_size, pin_memory):
    """
    Turns numpy samples and labels into DataLoader.
    """
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y).type(torch.LongTensor)

    dataset = TensorDataset(tensor_x, tensor_y)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory)
    return dataset_loader


def create_model(input_size, hidden_sizes, output_size):
    input_layer = [
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU()
    ]

    output_layer = [nn.Linear(hidden_sizes[-1], output_size)]

    hidden_layers = []
    for i in range(len(hidden_sizes) - 1):
        hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        hidden_layers.append(nn.ReLU())

    layers = tuple(input_layer + hidden_layers + output_layer)

    return Sequential(*layers)


def learn(index_picker, hidden_sizes, batch_size, learning_rate, n_epochs, data_files):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Runs on device: {}".format(device))

    # check if device is cuda
    if device.type == 'cuda':
        pin_memory = True
    else:
        pin_memory = False

    if data_files:
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_prepared_data(data_files)
    else:
        X, y = load_data()

        X = prepare_data(X, index_picker)

        X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, y)

    # prepare torch dataloaders
    train_loader = prepare_torch_data_loader(X_train, y_train, batch_size, pin_memory)

    val_loader = prepare_torch_data_loader(X_valid, y_valid, batch_size, pin_memory)

    test_loader = prepare_torch_data_loader(X_test, y_test, batch_size, pin_memory)

    # get number of classes
    n_classes = len(np.unique(y_train))

    # get number of features
    n_features = X_train.shape[1]

    # create model
    net = create_model(
        input_size=n_features,
        hidden_sizes=hidden_sizes,
        output_size=n_classes
    )
    print("Neural Network Architecture:")
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # fit model
    net.fit(train_loader, val_loader, criterion, optimizer, device, n_epochs)


if __name__ == "__main__":
    files = dict(
        X_train="X_train.npy",
        y_train="y_train.npy",
        X_valid="X_valid.npy",
        y_valid="y_valid.npy",
        X_test="X_test.npy",
        y_test="y_test.npy"
    )

    learn(
        index_picker=IndexPicker(25, 25),   # 25, 25
        hidden_sizes=[128, 128, 128],
        batch_size=512,
        learning_rate=0.0001,
        n_epochs=30,
        data_files=files
    )

    # X, y = load_data()
    #
    # X = prepare_data(X, IndexPicker(25, 25))
    #
    # X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X, y)
    #
    # np.save('X_train.npy', X_train)
    #
    # np.save('y_train.npy', y_train)
    #
    # np.save('X_valid.npy', X_valid)
    #
    # np.save('y_valid.npy', y_valid)
    #
    # np.save('X_test.npy', X_test)
    #
    # np.save('y_test.npy', y_test)
