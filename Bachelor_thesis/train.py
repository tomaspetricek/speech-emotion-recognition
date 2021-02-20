import numpy as np
import os
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from classifiers import Sequential
from pytorch_datasets import NumpySampleDataset, NumpyFrameDataset
from files import DatasetInfoFile, SetInfoFile

MODEL_DIR = "models/pytorch"


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

    PLOT_DPI = 200

    def __init__(self, model, train_dataset, val_dataset, test_datasets):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_datasets = test_datasets

    def __call__(self, batch_size, learning_rate, n_epochs, result_dir=None):
        if result_dir:
            self._set_logging(result_dir)

        # set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Runs on device: {}".format(device))

        # check if device is cuda
        if device.type == 'cuda':
            pin_memory = True
        else:
            pin_memory = False

        # prepare torch dataloaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory
        )

        print("Neural Network Architecture:")
        print(self.model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        print("Batch size: {}".format(batch_size))
        print("Learning rate: {}".format(learning_rate))

        # fit model
        history = self.model.fit(train_loader, self.val_dataset, self.test_datasets, criterion, optimizer, device,
                                 n_epochs)

        # get plots
        frame_acc_plot = self._get_plot(
            items=history["frame_acc"],
            title="model per frame accuracy",
            y_label="accuracy",
            x_label="epoch"
        )

        sample_acc_plot = self._get_plot(
            items=history["sample_acc"],
            title="model per sample accuracy",
            y_label="accuracy",
            x_label="epoch"
        )

        loss_plot = self._get_plot(
            items=history["loss"],
            title="model loss",
            y_label="loss",
            x_label="epoch"
        )

        # save results
        if result_dir:
            self._save_results(result_dir, frame_acc_plot, sample_acc_plot, loss_plot)

        # show results
        frame_acc_plot.show()
        sample_acc_plot.show()
        loss_plot.show()

    @staticmethod
    def _set_logging(result_dir):
        log_path = os.path.join(result_dir, "train.log")
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger('STDOUT')
        handler = logging.FileHandler(log_path, 'w')
        logging.StreamHandler.terminator = ""
        logger.addHandler(handler)
        sys.stdout.write = logger.info

    @staticmethod
    def _get_plot(items, title, y_label, x_label):
        plot = plt.figure()

        labels = list()
        for label, item in items.items():
            plt.plot(item)
            labels.append(label)

        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend(labels, loc='upper left')
        return plot

    def _save_results(self, result_dir, frame_acc_plot, sample_acc_plot, loss_plot):
        model_path = os.path.join(result_dir, "model.pt")
        torch.save(self.model, model_path)

        frame_acc_path = os.path.join(result_dir, 'frame_accuracy.png')
        frame_acc_plot.savefig(frame_acc_path, dpi=self.PLOT_DPI)

        sample_acc_path = os.path.join(result_dir, 'sample_accuracy.png')
        sample_acc_plot.savefig(sample_acc_path, dpi=self.PLOT_DPI)

        loss_path = os.path.join(result_dir, 'loss.png')
        loss_plot.savefig(loss_path, dpi=self.PLOT_DPI)


def prepare_dataset(directory, dataset_class, left_margin, right_margin, name=None):
    info_path = os.path.join(directory, "info.txt")
    n_samples, sample_lengths, sample_filename, label_filename = SetInfoFile(info_path).read()

    sample_path = os.path.join(directory, sample_filename)

    label_path = os.path.join(directory, label_filename)

    return dataset_class(n_samples, sample_lengths, sample_path, label_path, left_margin, right_margin, name)


def main(result_dir):
    dataset_dir = "prepared_data/en-4-re-90-10"

    left_margin = right_margin = 25

    info_path = os.path.join(dataset_dir, "info.txt")
    n_features, n_classes, n_samples = DatasetInfoFile(info_path).read()

    train_dir = os.path.join(dataset_dir, "train")
    train_dataset = prepare_dataset(train_dir, NumpyFrameDataset, left_margin, right_margin, name="inter")

    val_dir = os.path.join(dataset_dir, "test")
    val_dataset = prepare_dataset(val_dir, NumpySampleDataset, left_margin, right_margin, name="inter")

    test_dir = "prepared_data/it-4-re/whole"
    test_dataset_it = prepare_dataset(test_dir, NumpySampleDataset, left_margin, right_margin, name="ital")

    test_dir = "prepared_data/cz-4-re/whole"
    test_dataset_cz = prepare_dataset(test_dir, NumpySampleDataset, left_margin, right_margin, name="czech")

    input_size = n_features * (left_margin + 1 + right_margin)
    model = create_model(
        input_size=input_size,
        hidden_sizes=[128, 128, 128],
        output_size=n_classes
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_datasets=(test_dataset_it, test_dataset_cz)
    )

    result_dir = os.path.join(MODEL_DIR, result_dir)
    os.mkdir(result_dir)

    trainer(batch_size=512, learning_rate=0.0001, n_epochs=30, result_dir=result_dir)


if __name__ == "__main__":
    experiment_id = "exp_15"
    main(experiment_id)
