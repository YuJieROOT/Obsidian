基于PyTorch的简单神经网络，用于训练和测试MNIST手写数字分类模型。

### 导入库

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
```
- `DataLoader`: 用于加载数据集，支持批处理。
- `transforms`: 用于数据预处理，主要是将图像转换为 Tensor。
- `MNIST`: 一个常用的手写数字数据集。
- `matplotlib.pyplot`: 用于绘图，展示结果。

### 定义神经网络模型

```python
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x
```

这是一个简单的全连接神经网络，包含四层：

- `fc1`：输入层，输入尺寸为 28x28 的图像展平后是784个特征，输出64个节点。
- `fc2` 和 `fc3`：隐藏层，每层 64 个节点。
- `fc4`：输出层，输出 10 个节点，**对应 MNIST 的10个类别（0到9）。**

在`forward`方法中：

- 使用 `ReLU` 激活函数，非线性化数据。
- 最后一层使用`log_softmax`函数，将输出转换为对数概率，适用于分类任务。

### [[数据加载函数]]

```python
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)
```

`get_data_loader`函数用于加载 MNIST 数据集：

- `is_train` 参数决定是加载训练集还是测试集。
- 使用`transforms.ToTensor()`将图像转换为PyTorch 的 Tensor 格式。
- `DataLoader`用于分批加载数据，`batch_size=15`表示每个批次15个样本，`shuffle=True`表示每次加载时打乱数据顺序。

### [[模型评估函数]]

```python
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        # x = {Tensor: (15, 1,28, 28）}
        # y = {Tensor: (15,)}
        for (x, y) in test_data:
	        # outputs = {Tensor: (15,10)}
            outputs = net.forward(x.view(-1, 28*28))
            
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total
```

`evaluate`函数用于在测试集上评估模型的准确率：

- `test_data`是测试集，`net`是训练好的模型。
- 使用`torch.no_grad()`来关闭梯度计算，避免在评估时计算梯度。
- 计算每个样本的预测结果，并与实际标签进行比较，统计正确的预测数量，最后计算准确率。


### [[主函数]]

```python
def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    
    # 打印初始准确率
    print("initial accuracy:", evaluate(test_data, net))

	# 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
	
	# 训练网络循环
    for epoch in range(2):
        for (x, y) in train_data:
        # x = {Tensor: (15, 1,28, 28）}
        # y = {Tensor: (15,)}
            net.zero_grad()
            output = net.forward(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

	# 展示预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()
```

`main`函数是整个程序的执行入口，包含以下步骤：

1. 加载训练集和测试集。
2. 初始化神经网络（`net`）。
3. 输出初始化时的准确率。
4. 使用`Adam`优化器来优化网络的参数，学习率设为0.001。
5. 训练网络2个epoch，每个epoch都会遍历训练数据并更新网络权重。每个epoch结束后，输出当前测试集的准确率。
6. 训练完成后，展示前4张测试图像及其预测结果。

### 训练与测试流程

- 训练：通过`for (x, y) in train_data`遍历训练集，并在每个批次上计算损失并进行梯度更新。
- 测试：每个epoch后，通过`evaluate`函数计算当前模型在测试集上的准确率。
- 预测：训练完成后，展示一些测试集样本的预测结果，使用`matplotlib`绘制图像。

### 总结

这段代码展示了如何用PyTorch实现一个简单的神经网络来进行MNIST手写数字分类。通过定义一个简单的四层全连接网络，加载数据集，训练模型并评估模型的准确率。最终展示了模型对测试数据的预测结果。