from __future__ import print_function
import argparse
from atexit import register
from functools import reduce
from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from spikingjelly.activation_based import ann2snn, neuron
import numpy as np
from tqdm import tqdm
import sys
from .hooks import *

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(32, 10)
        )
        self.hooked = False

    def forward(self,x):
        x = self.network(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, loss_function):
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 在不计算梯度的情况下执行前向传播
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()  # 计算损失并累加
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率的索引
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)  # 计算平均损失

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def val(net, device, data_loader, T=None):
    correct = 0.0
    total = 0.0
    if T is not None:
        corrects = np.zeros(T)
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            if T is None:
                out = net(img)
                correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            else:
                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
                    corrects[t] += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
            break
    return correct / total if T is None else corrects / total

'''
def hook_val(net, device, data_loader, T=None):
    correct = 0.0
    total = 0.0
    if T is not None:
        corrects = np.zeros(T)
    hook_lists = register_hook(net, 'MNIST')
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            if T is None:
                out = net(img)
                correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            else:
                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
                    corrects[t] += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
            if net.hooked:
                unregister_hook(net, hook_lists)
                break
    return correct / total if T is None else corrects / total
'''

'''
def print_model_details(module, prefix=""):
    """
    递归打印模型的所有子模块及其详细信息
    """
    # 遍历当前模块的所有直接子模块
    for name, sub_module in module.named_children():
        breakpoint()
        # 创建一个表示当前层级的前缀
        new_prefix = prefix + ("  " if prefix else "") + name
        
        # 打印当前子模块的名称和其直接信息
        print(f"{new_prefix}: {sub_module}")
        
        # 递归调用，以打印更深层次的子模块
        print_model_details(sub_module, new_prefix)
'''

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(args.data_dir, train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST(args.data_dir, train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    if args.save_model:
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, loss_function)
            test(model, device, test_loader, loss_function)
            scheduler.step()

        torch.save(model.state_dict(), "mnist_cnn.pt")

    else:
        model.load_state_dict(torch.load('mnist_cnn.pt'))
    
    print('-------------------------------------------------')
    T = 50
    print('Converting using MaxNorm')
    model_converter = ann2snn.Converter(mode = 'max', dataloader = train_loader)
    snn_model = model_converter(model)
    # print(snn_model)
    # snn_model.print_readable()
    # print_model_details(snn_model)
    print('Simulating...')
    # 注册钩子
    # 存储hook句柄的列表，以便可以移除这些hook
    handles = []
    layerAttr = layer_attr()

    # 为模型的所有子模块添加hooks
    register_hook(snn_model, 'MNIST', handles)

    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(test_loader)):
            img = img.to(device)
            if T is None:
                out = snn_model(img)
            else:
                for m in snn_model.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = snn_model(img)
                        layerAttr.incStop()
                    else:
                        out += snn_model(img)
            print(label[:4])
            print(out.shape)
            print(out[:4])
            break
    
    print('-------------------------------------------------')
    print(layer_attr)
    # 如果需要在完成后清除hooks
    for handle in handles:
        handle.remove()
    # mode_max_accs = hook_val(snn_model, device, test_loader, T)
    # print('SNN accuracy (simulation %d time-steps): %.4f' % (T, mode_max_accs[-1]))


if __name__ == '__main__':
    main()
