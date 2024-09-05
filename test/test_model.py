# tests/test_model.py
import torch
from tqdm import tqdm

def train(model_dict, device, train_loader, optimizer, criterion, epoch):
    model = model_dict['model']
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, total, accuracy))
    return avg_loss

def test(model_dict, device, test_loader, criterion):
    model = model_dict['model']
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc='Testing')):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / total
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total, accuracy))
    return accuracy, test_loss


def test_model_performance(models, optimizers, criterion, train_loader, test_loader, device, num_epochs, model_names):
    """
    对每个模型执行训练和测试，并收集每个epoch的训练损失、测试损失和测试准确率。

    参数:
    ------
    models : list of nn.Module
        要训练的模型列表。
    optimizers : list of torch.optim.Optimizer
        每个模型对应的优化器列表。
    criterion : nn.Module
        损失函数。
    train_loader : DataLoader
        训练数据加载器。
    test_loader : DataLoader
        测试数据加载器。
    device : torch.device
        设备（CPU/GPU）。
    num_epochs : int
        训练轮数。
    model_names : list of str
        模型名称列表。

    返回:
    ------
    all_train_losses : list of lists
        每个模型的训练损失列表。
    all_test_losses : list of lists
        每个模型的测试损失列表。
    all_accuracies : list of lists
        每个模型的测试准确率列表。
    """
    all_train_losses = []
    all_test_losses = []
    all_accuracies = []

    for model, optimizer in zip(models, optimizers):
        train_losses = []
        test_losses = []
        accuracies = []
        epoch_list = []
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch} for {model_names[models.index(model)]}:")
            train_loss = train({'model': model, 'optimizer': optimizer, 'criterion': criterion}, device, train_loader,
                               optimizer, criterion, epoch)
            accuracy, test_loss = test({'model': model, 'optimizer': optimizer, 'criterion': criterion}, device,
                                       test_loader, criterion)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            epoch_list.append(epoch)
        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)
        all_accuracies.append(accuracies)

    return all_train_losses, all_test_losses, all_accuracies,epoch_list

