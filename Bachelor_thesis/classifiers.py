from typing import Any

from torch import nn
import torch


# class Classifier(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         n_input_neurons = 784
#         n_hidden_neurons = [300, 100]
#         n_output_neurons = 10
#
#         self.hidden_layer = nn.Linear(n_input_neurons, n_hidden_neurons)
#         self.output_layer = nn.Linear(n_hidden_neurons, n_output_neurons)
#
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, X):
#         X = self.hidden_layer(X)
#         X = self.sigmoid(X)
#         X = self.output_layer(X)
#         X = self.softmax(X)
#         return X


class Sequential(nn.Sequential):

    def _train(self, train_loader, optimizer, criterion):
        # set model to training mode
        self.train()

        correct = 0
        running_loss = 0.0
        n_samples = len(train_loader.dataset)

        for X, y in train_loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward propagation
            y_pred = self(X)

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

    def _eval(self, val_loader, criterion):
        # set module to evaluation mode
        self.eval()

        correct = 0
        running_loss = 0.0
        n_samples = len(val_loader.dataset)

        with torch.no_grad():
            for X, y in val_loader:
                # forward propagation
                y_pred = self(X)

                loss = criterion(y_pred, y)

                # statistics
                _, y_pred_label = torch.max(y_pred, 1)
                correct += (y_pred_label == y).sum().item()
                running_loss += loss.item() * X.shape[0]

        loss_ = running_loss / n_samples
        accuracy = correct / n_samples

        return loss_, accuracy

    def fit(self, train_loader, val_loader, criterion, optimizer, n_epochs=10):
        """
        Inspiration:
        https://nodata.science/no-fit-comparing-high-level-learning-interfaces-for-pytorch.html
        """
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")

            # train model
            train_loss, train_accuracy = self._train(train_loader, optimizer, criterion)

            # evaluate model
            val_loss, val_accuracy = self._eval(val_loader, criterion)

            # shows stats
            print(
                f"train_loss: {train_loss:0.3f} - train_accuracy: {train_accuracy:0.3f}",
                f" - val_loss: {val_loss:0.3f} - val_accuracy: {val_accuracy:0.3f}"
            )

        print('Finished Training')





