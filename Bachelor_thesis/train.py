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

    def __init__(self, model, train_dataset, val_dataset, test_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

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

        # fit model
        history = self.model.fit(train_loader, self.val_dataset, self.test_dataset, criterion, optimizer, device,
                                 n_epochs)

        # get plots
        accuracy_plot = self._get_accuracy_plot(history)
        loss_plot = self._get_loss_plot(history)

        # save results
        if result_dir:
            self._save_results(result_dir, accuracy_plot, loss_plot)

        # show results
        accuracy_plot.show()
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
    def _get_accuracy_plot(history):
        accuracy_plot = plt.figure()
        plt.plot(history['train_accuracy'])
        plt.plot(history['val_accuracy'])
        plt.plot(history['test_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        return accuracy_plot

    @staticmethod
    def _get_loss_plot(history):
        loss_plot = plt.figure()
        plt.plot(history['train_loss'])
        plt.plot(history['val_loss'])
        plt.plot(history['test_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        return loss_plot

    def _save_results(self, result_dir, accuracy_plot, loss_plot):
        model_path = os.path.join(result_dir, "model.pt")
        torch.save(self.model, model_path)

        accuracy_path = os.path.join(result_dir, 'accuracy.png')
        accuracy_plot.savefig(accuracy_path, dpi=self.PLOT_DPI)

        loss_path = os.path.join(result_dir, 'loss.png')
        loss_plot.savefig(loss_path, dpi=self.PLOT_DPI)


def prepare_dataset(directory, dataset_class, left_margin, right_margin):
    info_path = os.path.join(directory, "info.txt")
    n_samples, sample_lengths, sample_filename, label_filename = SetInfoFile(info_path).read()

    sample_path = os.path.join(directory, sample_filename)

    label_path = os.path.join(directory, label_filename)

    return dataset_class(n_samples, sample_lengths, sample_path, label_path, left_margin, right_margin)


def main(result_dir):
    dataset_dir = "prepared_data/fullset_npy_3"

    left_margin = right_margin = 30

    info_path = os.path.join(dataset_dir, "info.txt")
    n_features, n_classes, n_samples = DatasetInfoFile(info_path).read()

    train_dir = os.path.join(dataset_dir, "train")
    train_dataset = prepare_dataset(train_dir, NumpyFrameDataset, left_margin, right_margin)

    val_dir = os.path.join(dataset_dir, "val")
    val_dataset = prepare_dataset(val_dir, NumpySampleDataset, left_margin, right_margin)

    # test_dir = os.path.join(dataset_dir, "test")
    # test_dataset = prepare_dataset(test_dir, NumpySampleDataset, left_margin, right_margin)

    test_dir = "prepared_data/call_centers_npy/whole"
    test_dataset = prepare_dataset(test_dir, NumpySampleDataset, left_margin, right_margin)

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
        test_dataset=test_dataset
    )

    result_dir = os.path.join(MODEL_DIR, result_dir)
    os.mkdir(result_dir)

    trainer(batch_size=512, learning_rate=0.0001, n_epochs=30, result_dir=result_dir)


if __name__ == "__main__":
    experiment_id = "exp_07"
    main(experiment_id)
