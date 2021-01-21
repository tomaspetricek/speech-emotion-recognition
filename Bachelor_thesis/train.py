import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from classifiers import Sequential
from pytorch_datasets import CustomDataset


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


if __name__ == "__main__":

    train_dataset = CustomDataset(
        root_dir=,
        annotations_path=,
    )

    val_dataset = (
        root_dir=,
        annotations_path=,
    )

    test_datset = (
        root_dir=,
        annotations_path=,
    )

    model = create_model(
        input_size=n_features,
        hidden_sizes=[128, 128, 128],
        output_size=n_classes
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_datset
    )

    trainer(batch_size=512, learning_rate=0.0001, n_epochs=30)

