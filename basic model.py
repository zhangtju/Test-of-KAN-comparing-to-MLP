from kan import MultKAN
import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

class MNISTKANNetwork(nn.Module):
    def __init__(self, device='cpu'):
        super(MNISTKANNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.kan_layer1 = MultKAN(width=[[5, 0], [3, 0]], grid=5, k=3, noise_scale=0.5, grid_eps=0.02, grid_range=[-1, 1], device=device)
        self.kan_layer2 = MultKAN(width=[[3, 0], [2, 0]], grid=5, k=3, noise_scale=0.5, grid_eps=0.02, grid_range=[-1, 1], device=device)
        self.fc = nn.Linear(2, 10)  # 全连接层

    def forward(self, x):
        x = self.flatten(x)
        x = self.kan_layer1(x)
        x = self.kan_layer2(x)
        x = self.fc(x)  # 通过全连接层
        return x

print('==> Preparing data in classfication mission ......')
transform_mnist = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载用到的MNIST训练数据集
trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_mnist)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

# 加载MNIST测试数据集
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_mnist)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)



def train(model, device, train_loader, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
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

def test(model, device, test_loader, criterion):
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
    return accuracy

# 设置超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
num_epochs = 10
learning_rate = 0.001
# 初始化模型、优化器和损失函数
model = MNISTKANNetwork(device=device).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()
# 训练模型
for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}:")
    train(model, device, trainloader, optimizer, criterion)
    test(model, device, testloader, criterion)

