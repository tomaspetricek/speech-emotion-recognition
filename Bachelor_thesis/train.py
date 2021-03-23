import numpy as np
import os
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from classifiers import Sequential
from pytorch_datasets import NumpySampleDataset, NumpyFrameDataset
from files import DatasetInfoFile, SetInfoFile
from datasets import FOUR_EMOTIONS_VERBOSE, THREE_EMOTIONS_VERBOSE, ALL_EMOTIONS_VERBOSE

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

    model = Sequential(*layers)
    setattr(model, "n_classes", output_size)  # should be in Sequential constructor

    return model


class Trainer:
    PLOT_DPI = 200

    def __init__(self, model, train_dataset, val_dataset, test_datasets, classes_verbose):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_datasets = test_datasets
        self.classes_verbose = classes_verbose

    def __call__(self, batch_size, learning_rate, weight_decay=0., n_epochs=10, result_dir=None):
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
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        print("Batch size: {}".format(batch_size))
        print("Learning rate: {}".format(learning_rate))
        print("Weight decay: {}".format(weight_decay))

        # fit model
        history, conf_matrices = self.model.fit(train_loader, self.val_dataset, self.test_datasets, criterion,
                                                optimizer, device,
                                                n_epochs)

        # get plots
        frame_acc_plot = self._get_plot(
            items=history["frame_acc"],
            title="Přesnost modelu pro rámce",
            y_label="přesnost",
            x_label="epocha"
        )

        sample_acc_plot = self._get_plot(
            items=history["sample_acc"],
            title="Přesnost modelu pro náhrávky",
            y_label="přesnost",
            x_label="epocha"
        )

        loss_plot = self._get_plot(
            items=history["loss"],
            title="Ztráta modelu",
            y_label="ztráta",
            x_label="epocha"
        )

        conf_matrices_plots = []
        for label, conf_matrix in conf_matrices.items():
            title = "Matice záměn pro {}".format(label)
            plot = self._get_conf_matrix_plot(
                conf_matrix,
                title=title,
                y_label="předpovězené třídy",
                x_label="správné třídy",
            )
            conf_matrices_plots.append(plot)

        plots = [frame_acc_plot, sample_acc_plot, loss_plot] + conf_matrices_plots

        # save results
        if result_dir:
            self._save_results(result_dir, plots)

        for plot in plots:
            plot.show()

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

    def _get_conf_matrix_plot(self, conf_matrix, title, y_label, x_label):
        if self.classes_verbose:
            classes = self.classes_verbose
        else:
            classes = range(self.model.n_classes)
        df_cm = pd.DataFrame(conf_matrix, classes, classes)

        plot = plt.figure()
        sns.heatmap(df_cm, annot=True, fmt='g')

        plt.title(title)
        plt.xlabel(y_label)
        plt.ylabel(x_label)
        plt.tight_layout()

        return plot

    def _save_results(self, result_dir, plots):
        model_path = os.path.join(result_dir, "model.pt")
        torch.save(self.model, model_path)

        for plot in plots:
            title = plot.axes[0].get_title()
            filename = title + ".png"
            path = os.path.join(result_dir, filename)
            plot.savefig(path, dpi=self.PLOT_DPI)


def prepare_dataset(directory, dataset_class, left_margin, right_margin, name=None):
    info_path = os.path.join(directory, "info.txt")
    n_samples, sample_lengths, sample_filename, label_filename = SetInfoFile(info_path).read()

    sample_path = os.path.join(directory, sample_filename)

    label_path = os.path.join(directory, label_filename)

    return dataset_class(n_samples, sample_lengths, sample_path, label_path, left_margin, right_margin, name)


def main(result_dir):
    dataset_dir = "prepared_data/en-7-re-90-10"

    left_margin = right_margin = 15

    info_path = os.path.join(dataset_dir, "info.txt")
    n_features, n_classes, n_samples = DatasetInfoFile(info_path).read()

    train_dir = os.path.join(dataset_dir, "train")
    train_dataset = prepare_dataset(train_dir, NumpyFrameDataset, left_margin, right_margin, name="anglický")

    val_dir = os.path.join(dataset_dir, "test")
    val_dataset = prepare_dataset(val_dir, NumpySampleDataset, left_margin, right_margin, name="anglický")

    test_dir = "prepared_data/it-7-re/whole"
    test_dataset_it = prepare_dataset(test_dir, NumpySampleDataset, left_margin, right_margin, name="italský")

    test_dir = "prepared_data/cz-7-re/whole"
    test_dataset_cz = prepare_dataset(test_dir, NumpySampleDataset, left_margin, right_margin, name="český")

    test_datasets = [test_dataset_cz, test_dataset_it]

    input_size = n_features * (left_margin + 1 + right_margin)
    model = create_model(
        input_size=input_size,
        hidden_sizes=[128, 64],
        output_size=n_classes
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_datasets=test_datasets,
        classes_verbose=ALL_EMOTIONS_VERBOSE,
    )

    result_dir = os.path.join(MODEL_DIR, result_dir)
    os.mkdir(result_dir)

    trainer(batch_size=128, learning_rate=0.001, weight_decay=1e-4, n_epochs=10, result_dir=result_dir)


if __name__ == "__main__":
    experiment_id = "exp_24"
    main(experiment_id)
