import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize the linear classifier with logistic regression.

        Args:
            input_dim (int): Dimensionality of the input features.
        """
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (tensor): Input data tensor of shape (batch_size, input_dim).

        Returns:
            out (tensor): Output logits tensor of shape (batch_size, 1).
        """
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out




class SingleLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the one-hidden-layer neural network.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Number of neurons in the hidden layer.
        """
        super(SingleLayerNN, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (tensor): Input data tensor of shape (batch_size, input_dim).

        Returns:
            out (tensor): Output logits tensor of shape (batch_size, 1).
        """
        hidden = torch.relu(self.hidden_layer(x))
        out = self.output_layer(hidden)
        out = torch.sigmoid(out)
        return out
