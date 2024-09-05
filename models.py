# models.py
import torch
import torch.nn as nn
from kan import MultKAN

class MNISTKANNetwork(nn.Module):
    def __init__(self, device='cpu'):
        super(MNISTKANNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.kan_layer1 = MultKAN(width=[[2, 0],[5, 0],[1, 0]], grid=5, k=3, noise_scale=0.5, grid_eps=0.02, grid_range=[-1, 1], device=device)

        self.fc = nn.Linear(1, 1)  # 全连接层

    def forward(self, x):
        x = self.flatten(x)
        x = self.kan_layer1(x)
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self,inputlayer, device='cpu'):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(inputlayer, 16)  # 输入层到第一隐藏层
        self.fc2 = nn.Linear(16, 8)  # 第一隐藏层到第二隐藏层
        self.fc3 = nn.Linear(8, 1)  # 第二隐藏层到输出层

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层
        return x
