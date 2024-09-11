import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader , TensorDataset
import torchvision.transforms as transforms
import torchvision
import numpy as np


# 定义 RMSE 损失函数
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

def count_parameters(model):
    """
    计算给定模型中的可学习参数数量。
    参数:
    model (torch.nn.Module): PyTorch 模型实例。
    返回:
    int: 模型中的参数总数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_evaluate_models(models, optimizers, train_inputs, train_labels, test_inputs, test_labels, criterion,
                              epochs, model_names):
    """
    训练和评估多个模型，并返回每个模型的损失列表。

    参数:
    ------
    models : list of nn.Module
        要训练的模型列表。
    optimizers : list of torch.optim.Optimizer
        每个模型对应的优化器列表。
    train_inputs : torch.Tensor
        训练输入数据。
    train_labels : torch.Tensor
        训练标签数据。
    test_inputs : torch.Tensor
        测试输入数据。
    test_labels : torch.Tensor
        测试标签数据。
    criterion : nn.Module
        损失函数。
    epochs : int
        训练轮数。
    model_names : list of str
        模型名称列表。

    返回:
    ------
    loss_lists : list of lists
        每个模型的训练损失值列表。
    test_losses : list of lists
        每个模型的测试损失值列表。
    epoch_list : list
        每个 epoch 的编号列表。
    """
    loss_lists = [[] for _ in range(len(models))]
    test_losses = [[] for _ in range(len(models))]
    mae_lists = [[] for _ in range(len(models))]
    epoch_list = list(range(1, epochs + 1))

    # 训练模型
    for i, model in enumerate(models):
        for epoch in epoch_list:
            model.train()
            optimizers[i].zero_grad()
            outputs = model(train_inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizers[i].step()
            loss_lists[i].append(loss.item())
            if (epoch) % 1 == 0:
                print(f'Model{model_names[i]} - Epoch [{epoch}/{epochs}], Train Loss: {loss.item():.4f}')

            # 评估模型
            model.eval()
            with torch.no_grad():
                predictions = model(test_inputs)
                mae = torch.mean(torch.abs(predictions - test_labels))
                mae_lists[i].append(mae.item())
                test_loss = criterion(predictions, test_labels)
                test_losses[i].append(test_loss.item())
                print(f'Model{model_names[i]} - Epoch [{epoch}/{epochs}], Test Loss: {test_loss.item():.4f}, MAE: {mae.item()}')

    return loss_lists, test_losses, epoch_list


def plot_loss_vs_epoch(loss_lists, test_losses, epoch_list, model_names):
    """
    绘制每个模型的训练和测试损失随 epoch 变化的折线图。

    参数:
    ------
    loss_lists : list of lists
        每个模型的训练损失值列表。
    test_losses : list of lists
        每个模型的测试损失值列表。
    epoch_list : list
        每个 epoch 的编号列表。
    model_names : list of str
        模型名称列表。
    """
    # 颜色列表
    colors = ['b', 'r', 'g', 'm', 'c', 'k', 'y']
    # 绘制每个模型的损失曲线
    for i, (loss_list, test_loss_list) in enumerate(zip(loss_lists, test_losses)):
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_list, loss_list, marker='o', linestyle='-', color=colors[i % len(colors)], label='Train Loss')
        plt.plot(epoch_list, test_loss_list, marker='s', linestyle='--', color=colors[(i + 1) % len(colors)],
                 label='Test Loss')
        # 生成只包含每隔五个 epoch 的列表(epcoh很大的时候选择）
        # tick_indices = np.arange(0, len(epoch_list), 1)
        # tick_labels = [epoch_list[int(idx)] for idx in tick_indices]
        # plt.xticks(tick_indices, tick_labels)
        #epoch 小的时候选择
        plt.xticks(epoch_list)
        # plt.xticks(epoch_list)
        min_loss = min(min(loss_list), min(test_loss_list))
        max_loss = max(max(loss_list), max(test_loss_list))

        # 设置纵坐标刻度
        yticks_step = 0.1
        yticks_min = round(min_loss / yticks_step) * yticks_step
        yticks_max = round(max_loss / yticks_step) * yticks_step + yticks_step
        plt.yticks(ticks=np.arange(yticks_min, yticks_max + yticks_step, yticks_step))
        plt.title(f'Loss vs Epoch for {model_names[i]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_accuracy(accuracy_list,epoch_list,model_names):
    for i, accuracy_list in enumerate(accuracy_list):
      plt.figure(figsize=(10, 6))
      plt.plot(epoch_list, accuracy_list, marker='o', linestyle='-', label='accuracy')
      plt.xticks(epoch_list)
      plt.title(f'accuracy vs Epoch for {model_names[i]}')
      plt.xlabel('Epoch')
      plt.ylabel('accuracy')
      plt.legend()
      plt.grid(True)
      plt.show()





transform_mnist = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])
def get_train_loader(batch_size):
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_mnist)
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

def get_test_loader(batch_size):
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_mnist)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)