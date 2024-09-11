# main.py
import torch
import torch.optim as optim
import torch.nn as nn
from models import MNISTKANNetwork
from models import MLP
from test.test_model import test_model_performance
from tqdm import tqdm
from utils import *
from kan.utils import create_dataset
from kan.feynman import get_feynman_dataset


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_names=["MLP","KAN"]

    is_regression = True  # False for classfication

    if is_regression:
        print("REGRESION PORBLEM")

        # feyman data
        problem_id = 5  # problem_id in 1-120
        input_variables, expr, f, ranges = get_feynman_dataset(problem_id)
        n_var = len(input_variables)
        dataset = create_dataset(f, n_var, device='cpu')
        train_inputs = dataset['train_input']
        train_labels = dataset['train_label']
        test_inputs = dataset['test_input']
        test_labels = dataset['test_label']
        # 查看训练输入数据的范围
        train_inputs_min = torch.min(train_inputs, dim=0).values
        train_inputs_max = torch.max(train_inputs, dim=0).values
        print("Train Inputs Min:", train_inputs_min)
        print("Train Inputs Max:", train_inputs_max)
        # 查看测试输入数据的范围
        test_inputs_min = torch.min(test_inputs, dim=0).values
        test_inputs_max = torch.max(test_inputs, dim=0).values
        print("Test Inputs Min:", test_inputs_min)
        print("Test Inputs Max:", test_inputs_max)
        # 查看训练标签数据的范围
        train_labels_min = torch.min(train_labels)
        train_labels_max = torch.max(train_labels)
        print("Train Labels Min:", train_labels_min.item())
        print("Train Labels Max:", train_labels_max.item())
        # 查看测试标签数据的范围
        test_labels_min = torch.min(test_labels)
        test_labels_max = torch.max(test_labels)
        print("Test Labels Min:", test_labels_min.item())
        print("Test Labels Max:", test_labels_max.item())

        # define
        models = [MLP(inputlayer=n_var, device=device).to(device),
                  MNISTKANNetwork(device=device).to(device)]
        optimizers = [optim.SGD(model.parameters(), lr=0.001, momentum=0.9) for model in models]
        # criterion = nn.MSELoss()
        criterion = RMSELoss()
        epochs = 60

        # calculate para num
        param_counts = [count_parameters(model) for model in models]
        for i, param_count in enumerate(param_counts):
            print(f"Model{model_names[i]} has {param_count} trainable parameters.")

        # train and plot
        loss_lists, test_losses, epoch_list = train_and_evaluate_models(models, optimizers, train_inputs, train_labels, test_inputs, test_labels, criterion,
                              epochs,model_names)

        plot_loss_vs_epoch(loss_lists,test_losses,epoch_list,model_names)


    else:
        # MNIST
        print("CLASSIFICATION PROBLEM")
        # 定义两个模型
        models = [MLP(inputlayer=28 * 28, device=device).to(device),
                  MNISTKANNetwork(device=device).to(device)]

        # 定义优化器
        optimizers = [optim.SGD(model.parameters(), lr=0.001, momentum=0.9) for model in models]

        # 计算参数数量
        param_counts = [count_parameters(model) for model in models]
        for i, param_count in enumerate(param_counts):
            print(f"Model{model_names[i]} has {param_count} trainable parameters.")

        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 获取数据加载器
        train_loader = get_train_loader(batch_size=128)
        test_loader = get_test_loader(batch_size=100)
        num_epochs = 10

        trainloss_list,testloss_list,accuracy,epoch_list = test_model_performance(models, optimizers, criterion,train_loader,
                           test_loader, device ,num_epochs, model_names)
        plot_loss_vs_epoch(trainloss_list, testloss_list, epoch_list,model_names)
        plot_accuracy(accuracy,epoch_list,model_names)





