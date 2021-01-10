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


def load_dataset():
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
    X = np.array(list(samples['coefficients']))

    y = np.array(list(samples['emotion']))

    return X, y


def prepare_torch_dataset(batch_size, X, y):
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y).type(torch.LongTensor)

    dataset = TensorDataset(tensor_x, tensor_y)
    dataset_loader = DataLoader(dataset, batch_size=batch_size)
    return dataset, dataset_loader


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


def learn(index_picker, hidden_sizes, batch_size, learning_rate, n_epochs):
    # load datasets
    X, y = load_dataset()

    # get number of classes
    n_classes = len(np.unique(y))

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Runs on device: {}".format(device))

    # show cuda info
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # add margin
    X_margined = np.array(add_margin(X, index_picker))

    # reshape
    n_samples, window_length, n_features = X_margined.shape
    X_reshaped = np.array(np.reshape(X_margined, (n_samples, -1)))

    # split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_reshaped,
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

    # prepare torch datasets
    trainset, train_loader = prepare_torch_dataset(batch_size, X_train, y_train)

    valset, val_loader = prepare_torch_dataset(batch_size, X_valid, y_valid)

    testset, test_loader = prepare_torch_dataset(batch_size, X_test, y_test)

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
    learn(
        index_picker=IndexPicker(1, 1),   # 25, 25
        hidden_sizes=[128, 128, 128],
        batch_size=512,
        learning_rate=0.0001,
        n_epochs=5
    )
