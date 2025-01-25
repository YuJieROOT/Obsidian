# 定义模型
[[定义模型详解]]

```python
import torch.nn as nn  
  
class SimpleNet(nn.Module):  
    def __init__(self):  
        super(SimpleNet, self).__init__()  
        self.fc1 = nn.Linear(784, 256)  # 输入层到隐藏层  
        self.fc2 = nn.Linear(256, 10)   # 隐藏层到输出层  
  
    def forward(self, x):  
        x = torch.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x  
  
model = SimpleNet()
```

# 模型参数与设备转移

将模型 `model` 移动到 GPU（如果可用）或者 CPU 上进行计算

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model.to(device)  # 将模型移动到 GPU（如果可用）
```


# 数据处理
[[数据处理代码详解]]

```python
from torch.utils.data import Dataset, DataLoader  
  
# 假设 data 和 labels 是你的数据集和标签  
data = [[1, 2], [3, 4], [5, 6]]  # 示例数据  
labels = [0, 1, 0]  # 示例标签  
  
class CustomDataset(Dataset):  
    def __init__(self, data, labels):  
        self.data = data  
        self.labels = labels  
  
    def __len__(self):  
        return len(self.data)  
  
    def __getitem__(self, idx):  
        return self.data[idx], self.labels[idx]  
  
# 创建数据集实例  
dataset = CustomDataset(data, labels)  
  
# 创建 DataLoader 实例  
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  
  
# 测试 DataLoader
for batch in dataloader:  
    print(batch)

```



# 训练
[[训练代码详解 - 1]]

```python
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（分类任务）  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器
```

[[训练代码详解 - 2]]

```python
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  
  
# 假设 data 和 labels 是你的数据集和标签  
data = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)  # 示例数据，转换为张量  
labels = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)  # 示例标签，转换为张量  
  
# 自定义数据集  
class CustomDataset(Dataset):  
    def __init__(self, data, labels):  
        self.data = data  
        self.labels = labels  
  
    def __len__(self):  
        return len(self.data)  
  
    def __getitem__(self, idx):  
        return self.data[idx], self.labels[idx]  # 返回张量  
  
# 创建数据集实例  
dataset = CustomDataset(data, labels)  
  
# 创建 DataLoader 实例  
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  
  
# 定义一个线性模型
class TinyModel(nn.Module):  
    def __init__(self):  
        super(TinyModel, self).__init__()  
        self.linear = nn.Linear(1, 1)  # 输入维度 1，输出维度 1  
    def forward(self, x):  
        return self.linear(x)  
  
# 定义设备  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# 初始化模型、损失函数和优化器  
model = TinyModel().to(device)  # 将模型移动到设备  
criterion = nn.MSELoss()  # 均方误差损失函数  
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器  
  
# 训练循环  
for epoch in range(10):  
    model.train()  # 设置为训练模式  
    for inputs, labels in dataloader:  
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到设备  
  
        # 前向传播  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
  
        # 反向传播与优化  
        optimizer.zero_grad()  # 清空梯度  
        loss.backward()        # 计算梯度  
        optimizer.step()       # 更新参数  
  
    print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")  
  
# 测试模型  
model.eval()  # 设置为评估模式  
with torch.no_grad():  
    test_input = torch.tensor([[5.0]], dtype=torch.float32).to(device)  
    predicted_output = model(test_input)  
    print(f"Predicted output for input 5.0: {predicted_output.item():.4f}")
```