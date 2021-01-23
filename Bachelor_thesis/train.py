import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from classifiers import Sequential
from pytorch_datasets import NumpyDataset, NumpySplitDataset
from files import TextFile, DatasetInfoFile


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

def prepare_dataset(directory):
    info_path = os.path.join(directory, "info.txt")
    chunk_sizes, samples_filenames, labels_filenames = DatasetInfoFile(info_path).read()

    samples_paths = []
    for samples_filename in samples_filenames:
        samples_path = os.path.join(directory, samples_filename)
        samples_paths.append(samples_path)

    labels_paths = []
    for labels_filename in labels_filenames:
        label_path = os.path.join(directory, labels_filename)
        labels_paths.append(label_path)

    return NumpySplitDataset(samples_paths, labels_paths, chunk_sizes)


if __name__ == "__main__":

    dataset_dir = "prepared_data/fullset_npy_split"

    # train_dir = os.path.join(dataset_dir, "train")
    # train_samples_path = os.path.join(train_dir, "samples.npy")
    # train_labels_path = os.path.join(train_dir, "labels.npy")
    #
    # train_dataset = NumpyDataset(train_samples_path, train_labels_path)
    #
    # val_dir = os.path.join(dataset_dir, "val")
    # val_samples_path = os.path.join(val_dir, "samples.npy")
    # val_labels_path = os.path.join(val_dir, "labels.npy")
    #
    # val_dataset = NumpyDataset(val_samples_path, val_labels_path)
    #
    # test_dir = os.path.join(dataset_dir, "test")
    # test_samples_path = os.path.join(test_dir, "samples.npy")
    # test_labels_path = os.path.join(test_dir, "labels.npy")
    #
    # test_dataset = NumpyDataset(test_samples_path, test_labels_path)

    info_path = os.path.join(dataset_dir, "info.txt")
    info = TextFile(info_path).read_lines()

    n_features = int(info[0])
    n_classes = int(info[1])
    n_samples = int(info[2])

    train_dir = os.path.join(dataset_dir, "train")
    train_dataset = prepare_dataset(train_dir)

    val_dir = os.path.join(dataset_dir, "val")
    val_dataset = prepare_dataset(val_dir)

    test_dir = os.path.join(dataset_dir, "test")
    test_dataset = prepare_dataset(test_dir)

    model = create_model(
        input_size=n_features,
        hidden_sizes=[128, 128, 128],
        output_size=n_classes
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )

    trainer(batch_size=512, learning_rate=0.0001, n_epochs=30)

