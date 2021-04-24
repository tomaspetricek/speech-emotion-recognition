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
from collections import defaultdict
from classifiers import FeedForwardNet

from classifiers import Sequential
from datasets import NumpySampleDataset, NumpyFrameDataset
from files import DatasetInfoFile, SetInfoFile
from data import FOUR_EMOTIONS_VERBOSE, THREE_EMOTIONS_VERBOSE, ALL_EMOTIONS_VERBOSE

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

class Stats:
    def __init__(self, dataset_names):
        self.dataset_names = dataset_names
        self.sample_accuracies = defaultdict(list)
        self.frame_accuracies = defaultdict(list)
        self.losses = defaultdict(list)
        self.conf_matrices = dict()
        self.printer = StatsPrinter(self)

    def append(self, dataset_name, frame_acc, loss, sample_acc=None, conf_matrix=None):
        self.frame_accuracies[dataset_name].append(frame_acc)
        self.losses[dataset_name].append(loss)

        if sample_acc is not None:
            self.sample_accuracies[dataset_name].append(sample_acc)

        if conf_matrix is not None:
            self.conf_matrices[dataset_name] = conf_matrix

    def print_last_epoch(self):
        self.printer.print_last_epoch()

    def save(self, filename):
        stats = pd.DataFrame()

        for dataset_name in self.dataset_names:
            losses = self.losses[dataset_name]
            loss_col = "losses {}".format(dataset_name)
            stats[loss_col] = losses

            sample_accuracies = self.sample_accuracies[dataset_name]

            if sample_accuracies:
                samples_acc_col = "sample acc {}".format(dataset_name)
                stats[samples_acc_col] = sample_accuracies

            frame_accuracies = self.frame_accuracies[dataset_name]
            frame_acc_col = "frame acc {}".format(dataset_name)
            stats[frame_acc_col] = frame_accuracies

        stats.to_csv(filename, index=False)


class StatsPrinter:
    def __init__(self, stats):
        self.stats = stats
        self.header = f"{'epoch':^16}|{'name':^16}|{'loss':^16}|{'acc_frames':^16}|{'acc_samples':^16}"
        self.divider = "-" * len(self.header)
        self.header_printed = False

    def print_last_epoch(self):
        if not self.header_printed:
            print(self.divider)
            print(self.header)
            print(self.divider)
            self.header_printed = True

        for dataset_name in self.stats.dataset_names:
            loss = self.stats.losses[dataset_name]
            acc_frames = self.stats.frame_accuracies[dataset_name]
            acc_samples = self.stats.sample_accuracies[dataset_name]

            epoch = len(self.stats.losses[dataset_name])
            print(f"{epoch:^16}|{dataset_name:^16}|{loss[-1]:^16.3f}|{acc_frames[-1]:^16.3f}|", end="")

            if acc_samples:
                print(f"{acc_samples[-1]:^16.3f}")
            else:
                print()

        print(self.divider)

class Results:
    PLOT_DPI = 200

    def __init__(self, stats, classes_verbose):
        self.stats = stats
        self.classes_verbose = classes_verbose

        self.frame_acc_fig = self._plot(
            items=self.stats.frame_accuracies,
            title="Přesnost modelu pro vzorky",
            y_label="přesnost",
            x_label="epocha"
        )

        self.sample_acc_fig = self._plot(
            items=self.stats.sample_accuracies,
            title="Přesnost modelu pro náhrávky",
            y_label="přesnost",
            x_label="epocha"
        )

        self.loss_fig = self._plot(
            items=self.stats.losses,
            title="Ztráta modelu",
            y_label="ztráta",
            x_label="epocha"
        )

        self.conf_matrices_figs = []
        for label, conf_matrix in self.stats.conf_matrices.items():
            title = "Matice záměn pro {}".format(label)
            fig = self._plot_conf_matrix(
                conf_matrix,
                title=title,
                y_label="předpovězené třídy",
                x_label="správné třídy",
            )
            self.conf_matrices_figs.append(fig)

    def show(self):
        figs = [self.frame_acc_fig, self.sample_acc_fig, self.loss_fig] + self.conf_matrices_figs

        for plot in figs:
            plot.show()

    def _plot(self, items, title, y_label, x_label):
        fig = plt.figure()

        labels = list()
        for label, item in items.items():
            plt.plot(item)
            labels.append(label)

        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend(labels, loc='upper left')
        return fig

    def _plot_conf_matrix(self, conf_matrix, title, y_label, x_label):
        classes = self.classes_verbose
        df_cm = pd.DataFrame(conf_matrix, classes, classes)

        fig = plt.figure()
        sns.heatmap(df_cm, annot=True, fmt='g')

        plt.title(title)
        plt.xlabel(y_label)
        plt.ylabel(x_label)
        plt.tight_layout()

        return fig

    def save(self, dirname):
        figs = [self.frame_acc_fig, self.sample_acc_fig, self.loss_fig] + self.conf_matrices_figs

        for fig in figs:
            title = fig.axes[0].get_title()
            filename = title + ".png"
            path = os.path.join(dirname, filename)
            fig.savefig(path, dpi=self.PLOT_DPI)

class Trainer:

    def __init__(self, model, train_loader, val_dataset, test_datasets, optimizer, criterion, device, stats):
        self.model = model
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.test_datasets = test_datasets
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.stats = stats

    def _train(self, train_loader):
        # set model to training mode
        self.model.train()

        correct_frames = 0
        running_loss = 0.0
        n_frames = len(train_loader.dataset)

        for frames, labels in train_loader:
            # move data to device
            frames, labels = frames.to(self.device), labels.to(self.device)

            # make labels one dimensional
            labels = labels.flatten()

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward propagation
            pred = self.model(frames.float())

            loss = self.criterion(pred, labels)

            # back propagation
            loss.backward()

            self.optimizer.step()

            # add n correct frames
            _, pred_labels = torch.max(pred, 1)
            correct_frames += (pred_labels == labels).sum().item()

            # calc running loss
            batch_size = frames.shape[0]
            running_loss += loss.item() * batch_size

        loss_ = running_loss / n_frames
        accuracy_frames = correct_frames / n_frames

        self.stats.append(train_loader.dataset.name, accuracy_frames, loss_)

        return loss_, accuracy_frames

    # per sample accuracy
    def _validate(self, dataset):
        conf_matrix = np.zeros((self.model.n_classes, self.model.n_classes))
        # set module to evaluation mode
        self.model.eval()

        correct_samples = correct_frames = 0
        running_loss = 0.0
        n_samples = len(dataset)
        n_frames = dataset.n_frames

        with torch.no_grad():
            for frames, labels in dataset:
                # move data to device
                frames, labels = frames.to(self.device), labels.to(self.device)

                # make labels one dimensional
                labels = labels.flatten()

                # forward propagation
                pred = self.model(frames.float())

                loss = self.criterion(pred, labels)

                # add correct sample
                mean_pred = torch.mean(pred, 0)
                _, pred_label = torch.max(mean_pred, 0)
                correct_label = labels[0]

                if pred_label == correct_label:
                    correct_samples += 1

                conf_matrix[correct_label, pred_label] += 1

                # add n correct frames
                _, pred_labels = torch.max(pred, 1)
                correct_frames += (pred_labels == labels).sum().item()

                # calc running loss
                batch_size = frames.shape[0]
                running_loss += loss.item() * batch_size

        loss_ = running_loss / n_frames
        accuracy_samples = correct_samples / n_samples
        accuracy_frames = correct_frames / n_frames

        self.stats.append(dataset.name, accuracy_frames, loss_, accuracy_samples, conf_matrix)

        return loss_, accuracy_frames, accuracy_samples, conf_matrix

    def __call__(self, n_epochs=10):
        # move model to device
        self.model.to(self.device)

        for epoch in range(1, n_epochs + 1):
            # train model
            self._train(self.train_loader)

            self._validate(self.val_dataset)

            # test model
            for test_dataset in self.test_datasets:
                self._validate(test_dataset)

            self.stats.print_last_epoch()


def begin_logging(filename):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('STDOUT')
    handler = logging.FileHandler(filename, 'w')
    logging.StreamHandler.terminator = ""
    logger.addHandler(handler)
    sys.stdout.write = logger.info

def prepare_dataset(directory, dataset_class, left_margin, right_margin, name=None):
    info_path = os.path.join(directory, "info.txt")
    n_samples, sample_lengths, sample_filename, label_filename = SetInfoFile(info_path).read()

    sample_path = os.path.join(directory, sample_filename)

    label_path = os.path.join(directory, label_filename)

    return dataset_class(n_samples, sample_lengths, sample_path, label_path, left_margin, right_margin, name)


def main(result_dir):
    dataset_dir = "prepared_data/en-7-re-90-10"

    left_margin = right_margin = 25

    info_path = os.path.join(dataset_dir, "info.txt")
    n_features, n_classes, n_samples = DatasetInfoFile(info_path).read()

    train_dir = os.path.join(dataset_dir, "train")
    train_dataset = prepare_dataset(train_dir, NumpyFrameDataset, left_margin, right_margin, name="train: anglický")

    val_dir = os.path.join(dataset_dir, "test")
    val_dataset = prepare_dataset(val_dir, NumpySampleDataset, left_margin, right_margin, name="val: anglický")

    test_dir = "prepared_data/it-7-re/whole"
    test_dataset_it = prepare_dataset(test_dir, NumpySampleDataset, left_margin, right_margin, name="test: italský")

    test_datasets = [test_dataset_it]

    input_size = n_features * (left_margin + 1 + right_margin)

    model = FeedForwardNet(input_size, n_classes)

    # model_filename = os.path.join(MODEL_DIR, "exp_36", "model.pt")
    # model = torch.load(model_filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # check if device is cuda
    if device.type == 'cuda':
        pin_memory = True
    else:
        pin_memory = False

    batch_size = 32   # 128

    # prepare torch dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory
    )

    result_dirname = os.path.join(MODEL_DIR, result_dir)
    os.mkdir(result_dirname)

    learning_rate = 0.001
    weight_decay = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    dataset_names = [train_dataset.name, val_dataset.name, test_dataset_it.name]
    stats = Stats(dataset_names)

    log_filename = os.path.join(result_dirname, "train.log")
    begin_logging(log_filename)

    n_epochs = 10

    print("Optimizer: ", optimizer)
    print("Criterion: ", criterion)
    print("Model: ", model)
    print("N epochs:", n_epochs)
    print("Batch size: ", batch_size)
    print("Left margin: ", left_margin)
    print("Right margin: ", right_margin)

    trainer = Trainer(model, train_loader, val_dataset, test_datasets, optimizer, criterion, device, stats)
    trainer(n_epochs=n_epochs)

    model_filename = os.path.join(result_dirname, "model.pt")
    model.save(model_filename)

    result = Results(stats, FOUR_EMOTIONS_VERBOSE)
    result.show()
    result.save(result_dirname)

    stats_filename = os.path.join(result_dirname, "stats.csv")
    stats.save(stats_filename)


if __name__ == "__main__":
    experiment_id = "exp_12-4_emotions"
    main(experiment_id)
