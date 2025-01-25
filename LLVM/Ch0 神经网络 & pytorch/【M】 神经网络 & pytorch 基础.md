# 为什么选择 pytorch

- 训练速度快：`compiled_model = torch.compile（model）`，实现模型近一倍的提速
- 可用性强：从早期版本开始，API 一直保持一致。支持 CPU、GPU、并行处理以及分布式训练。
- 100％向后兼容：标志着用户可以大胆的进行版本升级
# pytorch 的下载安装

- 进入官网：[pytorch 下载链接 🔗](https://pytorch.org/get-started/locally/)
# 环境测试

```python
conda --version # 查看 conda 版本
conda env list # 查看已经创建的环境
conda create -n pytorch.cpu python==3.10 # 创建一个名字是 pytorch.cpu 的conda 环境
conda activate pytorch.cpu # 激活 pytorch.cpu 环境
conda list # 查看 pytorch.cpu 这个环境里面安装的包
python
import torch
torch.cuda.is_available()
```

# 神经网络


![[【G】 神经网络基础#^frame=uB9NZU5MzlP0zkZdGs0MK|神经网络结构]]
![[【G】 神经网络基础#^frame=ruxpwauHuaJtiL-Ex92J3|神经网络的表达式]]

# 损失函数
## 梯度下降
![[【G】 神经网络基础#^frame=wHDNz2F8fWIHL_oaymW7C|梯度下降]]

```ad-note
title: 如何得到损失函数最小的一个曲线，来拟合原曲线？
collapse: open
 
 $$ L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \tag{Loss Function}  $$

Q：如何让 Loss Function 最小？

A：
- 梯度下降
- 我们可以知道 $y_i = kx_i+b$，其实，就是找到一个最优的 $k、b$ 组合，使得 Loss Function 最小。即，问题转换为：两个参数，求极值的问题。
- **不能直接求导的原因**：参数过多，类似 $y_i = k_1x_1+k_2x_2+k_3x_3+\cdots+k_nx_n+b$，不容易求多参数的极值



```

## 常见的损失函数类型

| 损失函数类型    | 任务目标         | 典型应用场景         |
| :-------- | ------------ | -------------- |
| 交叉熵损失     | **分类**概率匹配   | 图像分类、文本分类      |
| MSE/MAE   | 连续值**预测**    | 回归分析、时间序列预测    |
| 对抗损失（GAN） | 生成数据分布匹配真实分布 | 图像生成、风格迁移      |
| 对比损失      | 学习相似性与差异性    | 自监督学习、跨模态检索    |
| 策略梯度损失    | 最大化累积奖励      | 强化学习（游戏、机器人控制） |
| 多任务加权损失   | 同时优化多个目标     | 目标检测、多模态任务     |

### 分类任务
- 损失函数：交叉熵损失 (Cross-Entropy Loss)
- 公式：
  $$ L = -\sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] $$

### 回归任务
- 损失函数：均方误差（MSE）、平均绝对误差（MAE）
- 公式：
  $$ L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$
$$ L_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| $$
  


# 如何建立模型
- 构建一个神经网络的结构本身
- 前向传播 ($x \rightarrow y$)
- 反向传播：建立真实的 $\hat y$ 和预测的 $y$ 之间的损失函数: $J(\hat y,y)$
	- 得到现在的网络和真实的差距是多少
	- 再根据损失函数，一步步迭代神经网络里面的每一个参数
- 评估：在一个测试集上验证构建的神经网络

# 手写数字体识别
![[【G】 神经网络基础#^frame=6N4SJuzVCpZJjBPRiSqVi|手写数字体识别]]