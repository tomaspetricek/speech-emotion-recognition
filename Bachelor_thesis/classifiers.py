from torch import nn
import torch
import numpy as np
from collections import defaultdict


class Sequential(nn.Sequential):

    def _train(self, train_loader, optimizer, criterion, device):
        # set model to training mode
        self.train()

        correct_frames = 0
        running_loss = 0.0
        n_frames = len(train_loader.dataset)

        for frames, labels in train_loader:
            # move data to device
            frames, labels = frames.to(device), labels.to(device)

            # make labels one dimensional
            labels = labels.flatten()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward propagation
            pred = self(frames.float())

            loss = criterion(pred, labels)

            # back propagation
            loss.backward()

            optimizer.step()

            # add n correct frames
            _, pred_labels = torch.max(pred, 1)
            correct_frames += (pred_labels == labels).sum().item()

            # calc running loss
            batch_size = frames.shape[0]
            running_loss += loss.item() * batch_size

        loss_ = running_loss / n_frames
        accuracy_frames = correct_frames / n_frames

        return loss_, accuracy_frames

    def _test(self, test_dataset, criterion, device):
        return self._eval(test_dataset, criterion, device)

    # per sample accuracy
    def _eval(self, val_dataset, criterion, device):
        # set module to evaluation mode
        self.eval()

        correct_samples = correct_frames = 0
        running_loss = 0.0
        n_samples = len(val_dataset)
        n_frames = val_dataset.n_frames

        with torch.no_grad():
            for frames, labels in val_dataset:
                # move data to device
                frames, labels = frames.to(device), labels.to(device)

                # make labels one dimensional
                labels = labels.flatten()

                # forward propagation
                pred = self(frames.float())

                loss = criterion(pred, labels)

                # add correct sample
                mean_pred = torch.mean(pred, 0)
                _, pred_label = torch.max(mean_pred, 0)
                correct_label = labels[0]

                if pred_label == correct_label:
                    correct_samples += 1

                # add n correct frames
                _, pred_labels = torch.max(pred, 1)
                correct_frames += (pred_labels == labels).sum().item()

                # calc running loss
                batch_size = frames.shape[0]
                running_loss += loss.item() * batch_size

        loss_ = running_loss / n_frames
        accuracy_samples = correct_samples / n_samples
        accuracy_frames = correct_frames / n_frames

        return loss_, accuracy_frames, accuracy_samples

    def fit(self, train_loader, val_dataset, test_datasets, criterion, optimizer, device, n_epochs=10):
        """
        Inspiration:
        https://nodata.science/no-fit-comparing-high-level-learning-interfaces-for-pytorch.html
        """

        # move model to device
        self.to(device)

        history = dict()
        sample_acc = defaultdict(list)
        frame_acc = defaultdict(list)
        loss = defaultdict(list)

        header = f"{'epoch':^16}|{'name':^16}|{'loss':^16}|{'acc_frames':^16}|{'acc_samples':^16}"
        divider = "-" * len(header)
        print(divider)
        print(header)
        print(divider)

        for epoch in range(n_epochs):
            print(divider)
            # train model
            train_loss, train_acc_frames = self._train(train_loader, optimizer, criterion, device)
            train_key = "train: {}".format(train_loader.dataset.name)
            loss[train_key].append(train_loss)
            frame_acc[train_key].append(train_acc_frames)
            print(f"{epoch:^16}|{train_key:^16}|{train_loss:^16.3f}|{train_acc_frames:^16.3f}")

            # evaluate model
            val_loss, val_acc_frames, val_acc_samples = self._eval(val_dataset, criterion, device)
            val_key = "val: {}".format(val_dataset.name)
            loss[val_key].append(val_loss)
            frame_acc[val_key].append(val_acc_frames)
            sample_acc[val_key].append(val_acc_samples)
            print(f"{epoch:^16}|{val_key:^16}|{val_loss:^16.3f}|{val_acc_frames:^16.3f}|{val_acc_samples:^16.3f}")

            # test model
            for test_dataset in test_datasets:
                test_loss, test_acc_frames, test_acc_samples = self._eval(test_dataset, criterion, device)
                test_key = "test: {}".format(test_dataset.name)

                loss[test_key].append(test_loss)
                frame_acc[test_key].append(test_acc_frames)
                sample_acc[test_key].append(test_acc_samples)
                print(f"{epoch:^16}|{test_key:^16}|{test_loss:^16.3f}|{test_acc_frames:^16.3f}|{test_acc_samples:^16.3f}")

        print('Finished Training')

        loss = dict(loss)
        frame_acc = dict(frame_acc)
        sample_acc = dict(sample_acc)
        history["loss"] = loss
        history["frame_acc"] = frame_acc
        history["sample_acc"] = sample_acc

        return history
