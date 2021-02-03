from torch import nn
import torch
import numpy as np


class Sequential(nn.Sequential):

    def _train(self, train_loader, optimizer, criterion, device):
        # set model to training mode
        self.train()

        correct = 0
        running_loss = 0.0
        n_samples = len(train_loader.dataset)

        for X, y in train_loader:
            # move data to device
            X, y = X.to(device), y.to(device)

            # make labels one dimensional
            y = y.flatten()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward propagation
            y_pred = self(X.float())

            loss = criterion(y_pred, y)

            # back propagation
            loss.backward()

            optimizer.step()

            # statistics
            _, y_pred_label = torch.max(y_pred, 1)
            correct += (y_pred_label == y).sum().item()
            running_loss += loss.item() * X.shape[0]

        loss_ = running_loss / n_samples
        accuracy = correct / n_samples

        return loss_, accuracy

    def _test(self, test_dataset, criterion, device):
        return self._eval(test_dataset, criterion, device)

    def _eval(self, val_dataset, criterion, device):
        # set module to evaluation mode
        self.eval()

        correct = 0
        running_loss = 0.0
        n_samples = len(val_dataset)

        with torch.no_grad():
            for X, y in val_dataset:
                # move data to device
                X, y = X.to(device), y.to(device)

                # make labels one dimensional
                y = y.flatten()

                # forward propagation
                y_pred = self(X.float())

                loss = criterion(y_pred, y)

                # calc mean label
                y_mean = torch.mean(y_pred, 0)
                _, pred_class = torch.max(y_mean, 0)

                correct_class = y[0]

                if pred_class == correct_class:
                    correct += 1

                running_loss += loss.item()

        loss_ = running_loss / n_samples
        accuracy = correct / n_samples

        return loss_, accuracy

    def fit(self, train_loader, val_dataset, test_dataset, criterion, optimizer, device, n_epochs=10):
        """
        Inspiration:
        https://nodata.science/no-fit-comparing-high-level-learning-interfaces-for-pytorch.html
        """

        # move model to device
        self.to(device)

        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")

            # train model
            train_loss, train_accuracy = self._train(train_loader, optimizer, criterion, device)

            # evaluate model
            val_loss, val_accuracy = self._eval(val_dataset, criterion, device)

            # test model
            test_loss, test_accuracy = self._eval(test_dataset, criterion, device)

            # shows stats
            print(
                f"train_loss: {train_loss:0.3f} - train_accuracy: {train_accuracy:0.3f}",
                f" - val_loss: {val_loss:0.3f} - val_accuracy: {val_accuracy:0.3f}",
                f" - test_loss: {test_loss:0.3f} - test_accuracy: {test_accuracy:0.3f}"
            )

        print('Finished Training')





