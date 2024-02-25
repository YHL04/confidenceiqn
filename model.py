
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Net(nn.Module):
    def __init__(self, dim_x=32, dim_y=32, channels=3, classes=100):
        super(Net, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*(dim_x-4)*(dim_y-4)//4, 128)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output


class ConfModelScalar(nn.Module):
    def __init__(self, dim_x=32, dim_y=32, channels=3):
        super(ConfModelScalar, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*(dim_x-4)*(dim_y-4)//4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)

        return output


class ConfModel(nn.Module):
    def __init__(self, dim_x=32, dim_y=32, channels=3):
        super(ConfModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.d_model = 128
        self.n_cos = 64

        self.conv1 = nn.Conv2d(channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*(dim_x-4)*(dim_y-4)//4, self.d_model)

        self.cos_embedding = nn.Linear(self.n_cos, self.d_model)
        self.linear = nn.Linear(self.d_model, self.d_model)
        self.out = nn.Linear(self.d_model, 1)

        self.pis = torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1, 1, self.n_cos).to(self.device)

        self.gelu = nn.GELU()

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the co-sin values depending on the number of tau samples
        """
        assert torch.equal(self.pis,
                           torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1, 1, self.n_cos).to(self.device))

        # (batch_size, n_tau, 1)
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).to(self.device)
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos)

        cos = cos.view(batch_size * n_tau, self.n_cos)
        cos = self.gelu(self.cos_embedding(cos))
        cos = cos.view(batch_size, n_tau, self.d_model)

        return cos, taus

    def forward(self, x, n_tau):
        """
        :param x:     Tensor[batch_size, 1, d_model]
        :param n_tau: int
        :return:      Tensor[batch_size, n_tau]
                      Tensor[batch_size, n_tau]
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        # IQN begins here
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        assert x.shape == (batch_size, 1, self.d_model)

        cos, taus = self.calc_cos(batch_size, n_tau)

        cos = cos.view(batch_size, n_tau, self.d_model)
        taus = taus.view(batch_size, n_tau)

        x = (x * cos).view(batch_size * n_tau, self.d_model)
        x = self.gelu(self.linear(x))
        x = self.out(x)
        x = x.view(batch_size, n_tau)

        return x, taus

