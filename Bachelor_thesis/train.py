import numpy as np
import os
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from classifiers import Sequential
from pytorch_datasets import NumpyDataset, NumpySplitDataset
from files import DatasetInfoFile, SetInfoFile
from tools import IndexPicker

ENABLE_LOGGING = True


if ENABLE_LOGGING:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('STDOUT')
    handler = logging.FileHandler('logging/train.log', 'w')
    logging.StreamHandler.terminator = ""
    logger.addHandler(handler)
    sys.stdout.write = logger.info


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


class Trainer:

    def __init__(self, model, train_dataset, val_dataset, test_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def __call__(self, batch_size, learning_rate, n_epochs):

        # set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Runs on device: {}".format(device))

        # check if device is cuda
        if device.type == 'cuda':
            pin_memory = True
        else:
            pin_memory = False

        # prepare torch dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, pin_memory=pin_memory)

        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, pin_memory=pin_memory)

        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, pin_memory=pin_memory)

        print("Neural Network Architecture:")
        print(self.model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # fit model
        self.model.fit(train_loader, val_loader, criterion, optimizer, device, n_epochs)


def prepare_dataset(directory, index_picker):
    info_path = os.path.join(directory, "info.txt")
    _, sample_lengths, sample_filename, label_filename = SetInfoFile(info_path).read()

    sample_path = os.path.join(directory, sample_filename)

    label_path = os.path.join(directory, label_filename)

    return NumpyDataset(sample_lengths, sample_path, label_path, index_picker)


if __name__ == "__main__":
    dataset_dir = "prepared_data/fullset_npy_3"

    index_picker = IndexPicker(25, 25)

    info_path = os.path.join(dataset_dir, "info.txt")
    n_features, n_classes, n_samples = DatasetInfoFile(info_path).read()

    train_dir = os.path.join(dataset_dir, "train")
    train_dataset = prepare_dataset(train_dir, index_picker)

    val_dir = os.path.join(dataset_dir, "val")
    val_dataset = prepare_dataset(val_dir, index_picker)

    test_dir = os.path.join(dataset_dir, "test")
    test_dataset = prepare_dataset(test_dir, index_picker)

    input_size = n_features * (index_picker.left_margin + 1 + index_picker.right_margin)
    model = create_model(
        input_size=input_size,
        hidden_sizes=[128, 128, 128],
        output_size=n_classes
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )

    trainer(batch_size=512, learning_rate=0.0001, n_epochs=5)
