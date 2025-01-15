
# 标题

- 针对的问题：无监督背光图像增强
- 提出的核心技术：迭代提示学习

# 摘要 
- 针对的问题：无监督逆光图像增强
- 研究的方法：对比语言-图像预训练 “Contrastive Language-Image Pre-Training (CLIP)” ([Liang 等, 2023, p. 8094](zotero://select/library/items/UQEAZEHS)) ([pdf](zotero://open-pdf/library/items/P5VKJI53?page=1&annotation=M9IDIRBT))
- 遇到的困难：很难找到准确的提示词用于图像增强任务
- 解决的方法：
	- 开源的 CLIP 先验促进增强网络的优化
		- **可以区分逆光图像和光照良好的图像**
		- 感知具有**不同亮度的异质区域**
	- <b><font color="#B22222">提示学习框架</font></b>
		- 约束 “text-image similarity” ([Liang 等, 2023, p. 8094](zotero://select/library/items/UQEAZEHS)) ([pdf](zotero://open-pdf/library/items/P5VKJI53?page=1&annotation=QQ9EYBXE)) 
		- 约束 CLIP **潜在空间**中对应图像（逆光图像/光照良好的图像）之间的关系
	- <b><font color="#B22222">训练增强网络</font></b>
	- 迭代微调提示学习框架 $\Rightarrow$ 减少 逆光图像、增强结果和光照良好图像之间的分布差异
	- 排序学习 $\Rightarrow$ 提升增强性能
	- 在提示学习框架和增强网络之间交替进行更新
- 效果：
	- 在**视觉质量**和**泛化能力**方面优于当前最先进的方法
	- 不需要任何配对数据【zero shot】


# 引言

- 提出问题及其原因
	- 手动校正背光图像很难
	- 在增强 曝光不足的区域 的同时 保留光线充足的区域 也很难
- 文献调研
    - 相关内容
    - 缺点批判
- 本文的贡献
# 问题建模
![[Pasted image 20250111160728.png]]
1）**提示初始化（Prompt Initialization）**  
	![[Pasted image 20250111161241.png]]
	我们首先通过**预训练的 CLIP 图像和文本编码器**，将逆光图像和光照良好图像 连同 可学习的提示对（正样本和负样本）编码到潜在空间中。通过缩小图像和文本在潜在空间中的距离，我们获得了一个：可以有效区分逆光图像和光照良好图像的**初始提示对**。

2）**基于 CLIP 的增强训练（CLIP-aware Enhancement Training）**  
	通过初始化的提示词，我们使用 CLIP 嵌入空间中的 文本-图像相似性 约束**训练一个增强网络**。

3）**提示优化（Prompt Refinement）**  
	我们引入了一种**提示微调机制**，通过[[排序学习]]进一步区分逆光图像、增强结果和光照良好图像之间的分布差异，从而更新提示词。我们迭代地更新增强网络和提示学习框架，直到获得视觉上令人满意的结果。

# 方法和分析

- 文字链接所有公式
- 逻辑清晰顺畅
- 适当的停顿和解释
    - 目前获得了什么意义
    - 能得到什么样的启示
    - 与其他的文章比较
    - 已经有什么样的好处
- 从读者角度出发（不要写得自己日后看不懂）
- 掌握一套数学分析工具（理论）

## 提示初始化

### **1. 输入初始化**
- 给定两种图像：
  1. **逆光图像** $I_b \in \mathbb{R}^{H \times W \times 3}$：光线条件较差的图像。
  2. **光线良好的图像** $I_w \in \mathbb{R}^{H \times W \times 3}$：作为参考图像，光线条件较好。
- 随机初始化两个提示（prompts）：
  1. **正向提示** $T_p \in \mathbb{R}^{N \times 512}$：用于表示光线良好的正样本信息。
  2. **负向提示** $T_n \in \mathbb{R}^{N \times 512}$：用于表示逆光图像的负样本信息。
  - **N** ：每个提示中嵌入标记的数量（可以理解为提示中的单词或标记数量）。
  - **512**：每个单元包含 512 维特征

---

### **2. 编码特征提取**
- 使用预训练的 **CLIP 模型**：
  1. **图像编码器** $\Phi_{image}$：对逆光图像 $I_b$ 和光线良好的图像 $I_w$ 进行编码，提取**潜在图像特征表示**。【latent code】
  2. **文本编码器** $\Phi_{text}$：对正向提示 $T_p$ 和负向提示 $T_n$ 进行编码，提取**潜在文本特征表示**。

---

### **3. 基于 文本-图像相似性 进行优化**
- **CLIP 潜在空间**中的特点是：图像编码和文本编码可以**通过余弦相似性进行对齐。**
- 目标是：通过学习，使正向提示 $T_p$ 更贴近光线良好的图像 $I_w$，负向提示 $T_n$ 更贴近逆光图像 $I_b$。

#### **分类预测**
1. **分类公式**：
   - 预测值 $\hat{y}$ 通过以下公式计算：
$$
     \hat{y} = \frac{e^{cos(\Phi_{image}(I), \Phi_{text}(T_p))}}{\sum_{i \in \{n, p\}} e^{cos(\Phi_{image}(I), \Phi_{text}(T_i))}}
$$
 - $cos(\cdot, \cdot)$：图像编码和文本编码之间的余弦相似度。
 - $I \in \{I_b, I_w\}$：当前输入的图像。
 - $T_p$：正向提示，$T_n$：负向提示。

```ad-note
title: 为什么分子中只有 $T_p$？
collapse: open

分子中只有 $T_p$ 是因为只在计算输入图像 $I$ 属于光线良好类的概率。

如果是光线良好的图输入，则希望结果是 $1$ 

如果是背光图像的输入，则希望结果是 $0$

```

2. **二元交叉熵损失**：
   - 使用 $\mathcal{L}_{initial}$ 对预测值 $\hat{y}$ 和**实际标签 $y$** 进行优化：
$$
     \mathcal{L}_{initial} = -(y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}))
$$
     
     - $y = 0$：当输入图像为负样本（逆光图像 $I_b$）。
     - $y = 1$：当输入图像为正样本（光线良好的图像 $I_w$）。
   - 目标是**最小化该损失函数**，从而让提示 $T_p$ 和 $T_n$ 能够正确区分光线良好图像和逆光图像。

---

### **总结**
- **输入：** 逆光图像 $I_b$ 和光线良好的图像 $I_w$。
- **初始化：** 随机生成正向提示 $T_p$ 和负向提示 $T_n$。
- **编码：** 使用 CLIP 的图像编码器和文本编码器提取潜在特征。
- **优化：** 
  1. 通过**余弦相似性**计算正向和负向提示与图像编码的相似度。
  2. 使用二元交叉熵损失，让提示 $T_p$ 与光线良好的图像 $I_w$ 更匹配，提示 $T_n$ 与逆光图像 $I_b$ 更匹配。

最终，**通过学习得到的提示 $T_p$ 和 $T_n$** 可以更好地表征图像的光线条件。

## 训练初始增强网络
![[Pasted image 20250112151606.png]]
### **1. 基本概念和任务**

目标是使用**CLIP感知损失**来训练一个增强网络，该网络可以增强逆光图像 $I_b$，生成光线更好的图像 $I_t$。这个过程基于从第一阶段获得的初始提示（正向提示 $T_p$ 和负向提示 $T_n$）。

1. **增强网络选择：**  
   - 使用一个**Unet**作为基线模型来增强逆光图像。
   - 同样，可以使用更复杂的网络架构。

2. **光照估计：**
   - 增强网络根据[[用于光线增强的Retinex模型]]  [[估计光照图]] $I_i \in \mathbb{R}^{H \times W \times 1}$。
   - 最终结果通过以下公式生成： $I_t = \frac{I_b}{I_i},$

其中 $I_t$ 是增强后的图像。

---

### **2. 损失函数设计**
增强网络的训练基于两个核心损失函数：**CLIP增强损失**和**恒等损失**。最终的总损失是它们的加权组合。

#### **2.1 CLIP增强损失** $\mathcal{L}_{clip}$

- **目的：** 衡量增强后的结果 $I_t$ 与 CLIP 空间中正向和负向提示之间的相似性。==希望增强图像 $I_t$ 更接近光线良好的图像（即更符合正向提示 $T_p$ 的描述）。==
- **计算公式：**
  
$$
  \mathcal{L}_{clip} = \frac{e^{cos(\Phi_{image}(I_t), \Phi_{text}(T_n))}}{\sum_{i \in \{n, p\}} e^{cos(\Phi_{image}(I_t), \Phi_{text}(T_i))}}
$$
  - $cos(\cdot, \cdot)$：CLIP 图像编码和文本编码之间的余弦相似度。
  - $T_n$ 和 $T_p$：分别是负向和正向提示。

---

#### **2.2 恒等损失** $\mathcal{L}_{identity}$
- ==**目的：** 保证增强图像 $I_t$ 在内容和结构上与原始逆光图像 $I_b$ 保持一致。==
- **计算公式：**
$$
  \mathcal{L}_{identity} = \sum_{l=0}^{4} \alpha_l \cdot ||\Phi_{image}^l(I_b) - \Phi_{image}^l(I_t)||_2
$$
  - $\Phi_{image}^l(\cdot)$：CLIP 图像编码器**在第 $l$ 层的特征输出。**
  - $\alpha_l$：每一层的权重，用于调节不同层次特征对内容和结构的影响。
```ad-note
title:这里的l为什么最大只到了4
collapse: open

- **为什么到  $l=4$ ？**
  1. CLIP 图像编码器基于 ResNet101，具有 5 个关键特征层（0 到 4）。
  2. 第  $l=4$  层是 ResNet 的最高语义特征层，表示全局信息，没有更高层的特征存在。
  
- **为什么第  $l=4$  层权重较低？**
  1. 第  $l=4$  层更关注颜色调整和全局语义，而增强任务需要保持图像内容和结构的一致性，因此赋予低权重（0.5）。  
  2. 低层和中层特征对图像的细节和局部信息有更大贡献，因此权重较高（1.0）。

- **实际意义：**
  这是对特征层次的合理利用，既保证了增强效果的语义一致性，又能保留原始图像的细节和结构信息。
```
#### **2.3 最终损失组合**
- 总损失函数是两种损失的加权组合：
$$
  
  \mathcal{L}_{enhance} = \mathcal{L}_{clip} + w \cdot \mathcal{L}_{identity}
  
$$
  - 权重 $w = 0.9$：用于平衡 CLIP 增强损失和恒等损失的影响。

---

### **3. 训练流程分阶段**
为了更好地优化增强网络，训练分为两个阶段：
#### **3.1 自重构阶段**
- **目的：** 使用恒等损失 $\mathcal{L}_{identity}$ 来确保增强结果在像素空间中与原始逆光图像 $I_b$ 保持一致。
- **设置：** 在公式 (4) 中，将所有层的权重 e 设置为 $1.0$，即：
$$
  
  \alpha_{l=0,1,\ldots,4} = 1.0
  
$$
#### **3.2 逆光增强阶段**
- **目的：** 综合使用恒等损失 $\mathcal{L}_{identity}$ 和 CLIP 增强损失 $\mathcal{L}_{clip}$ 对增强网络进行完整训练。
- **设置：**
  - 根据经验调整不同层的权重：
$$
    
    \alpha_{l=0,1,2,3} = 1.0, \quad \alpha_{l=4} = 0.5
    
$$
  - 原因：
    - 较低层$（l=0,1,2,3）$的特征与图像的结构和内容更相关，权重较高。
    - 最后一层$（l=4）$的特征与图像颜色更相关，由于我们希望调整颜色，权重较低。

---

### **总结**
- **输入：** 逆光图像 $I_b$，通过增强网络预测光照图 $I_i$ 并生成增强图像 $I_t = I_b / I_i$。
- **损失：** 结合 $\mathcal{L}_{clip}$（增强效果的感知损失）和 $\mathcal{L}_{identity}$（内容一致性损失）。
- **训练策略：**
  1. 首先进行自重构训练，专注于恒等损失。
  2. 然后综合使用两种损失进行完整训练，调整不同层的特征权重。
- **网络输出：** 一个在光线条件和视觉感知上更接近光线良好图像的增强结果。

**Input:** A backlit image  $I_b$ , where the illumination map  $I_i$  is predicted by the enhancement network, and the enhanced image is generated as  $I_t = I_b / I_i$ .
**Loss Function:** Combines  $\mathcal{L}_{clip}$  (perceptual loss for enhancement effect) and  $\mathcal{L}_{identity}$  (content consistency loss).
**Training Strategy:**
1. Perform self-reconstruction training first, focusing on **identity loss**.
2. Then, conduct full training **with both losses**, adjusting feature weights at different layers.
**Network Output:** An enhanced result that is closer to well-lit images in terms of lighting conditions and visual perception.


## **3.2 提示词优化与增强调优**

### 1. **目标**
- **主要目标**：通过 **提示词优化** 和 **网络调优** 的交替执行，提升提示词的区分能力。具体包括：
  - 识别逆光图像、增强结果和良好光照图像之间的差异。
  - 更好地感知不同亮度区域的异质性。

---

### 2. **提示词优化的重要性**
- **问题**：初始提示词（从逆光和良好光照图像生成）无法捕捉细粒度的光线和亮度差异。
- **解决方案**：进一步优化可学习的正提示词（描述增强目标）和负提示词（避免的特性）。
- **方法**：引入 [[边距排序损失]] 来更新提示词，从而引导增强结果逐步接近理想的良好光照图像。

---

### 3. **负相似度分数公式**
- 定义了图像与提示词对的**相似度**（通过负相似度分数 $S(I)$ 表示），公式为：
$$
S(I) = \frac{e^{\cos(\Phi_{\text{image}}(I), \Phi_{\text{text}}(T_n))}}{\sum_{i \in \{n, p\}} e^{\cos(\Phi_{\text{image}}(I), \Phi_{\text{text}}(T_i))}}
$$


**解释**：
- $\Phi_{\text{image}}$：表示图像在 CLIP 嵌入空间中的特征。
- $\Phi_{\text{text}}$：表示提示词在 CLIP 嵌入空间中的特征。
- $T_n$：负提示词；$T_p$：正提示词。
- **作用**：将增强后的图像特征与理想目标（正提示词）靠近，同时远离不理想目标（负提示词）。

---

### 4. **初始边距排序损失**
$$
\begin{aligned}
\mathcal{L}_{\text{prompt1}} & = \max\big(0, S(I_w) - S(I_b) + m_0\big) \\
& + \max\big(0, S(I_t) - S(I_b) + m_0\big) \\
& + \max\big(0, S(I_w) - S(I_t) + m_1\big)
\end{aligned}

$$


- **含义**：
  1. $S(I_w) - S(I_b) + m_0$：良好光照图像 $I_w$ 与逆光图像 $I_b$ 的分数差异应大于边距 $m_0$（目标：区分逆光与良好光照图像）。
  2. $S(I_t) - S(I_b) + m_0$：增强结果 $I_t$ 应远离逆光图像。
  3. $S(I_w) - S(I_t) + m_1$：增强结果 $I_t$ 应接近良好光照图像 $I_w$。
- **边距设置**：
  - $m_0 = 0.9$：确保逆光与良好光照图像间有显著分离。
  - $m_1 = 0.2$：使增强结果与良好光照图像靠近。

---

### 5. **引入历史增强结果的优化**
- **问题**：为了进一步提高提示词性能，需要**考虑增强网络的历史行为**。
- **解决**：引入上一轮增强结果 $I_{t-1}$，更新边距排序损失公式：
$$
\begin{aligned}
\mathcal{L}_{\text{prompt2}} &= \max(0, S(I_w) - S(I_b) + m_0) 
\\
& + \max(0, S(I_{t-1}) - S(I_b) + m_0) \\
& + \max(0, S(I_w) - S(I_t) + m_1) \\
& + \max(0, S(I_t) - S(I_{t-1}) + m_2)
\end{aligned}
$$


- **新增项**：
  - $S(I_{t-1}) - S(I_b) + m_0$：上一轮增强结果应远离逆光图像。
  - $S(I_t) - S(I_{t-1}) + m_2$：当前增强结果应接近上一轮增强结果。
- **边距设置**：
  - $m_2 = m_1 = 0.2$：保证每轮增强间**保持平滑的光线和色彩调整。**

---

### 6. **最终优化目标**
- 新的提示词不仅能更好地分辨图像的光线差异，还能平衡连续增强的平滑性和合理性。
- **图 10**：展示了提示词优化如何使增强结果更加关注图像光线和颜色，而非高级内容。

---

## 总结
### 目标
1. **区分能力**：准确区分逆光图像、增强结果和良好光照图像。
2. **细节感知**：更好地捕捉图像中不同亮度区域的细粒度差异。
### 主要方法：
1. **提示词优化**：
   - 针对初始提示词不足的问题，优化正提示词和负提示词。
   - 利用边距排序损失函数 $\mathcal{L}{\text{prompt1}}$，通过图像和提示词的相似性调整提示词，使其更准确地表示光照和颜色的分布。

2. **增强损失改进**：
   - 引入前一轮增强结果 $I_{t-1}$，在损失函数 $\mathcal{L}_{\text{prompt2}}$ 中加入约束。
   - 设置多重边距 $m_0, m_1, m_2$，确保增强结果与良好光照图像接近，同时保持与前一轮增强结果的相似性。

3. **交替迭代**：
   - 优化提示词后调整增强网络。
   - 使用增强结果进一步优化提示词，重复迭代以逐步提升性能。

**核心思想**：通过优化提示词和增强网络的协同作用，使模型能够学习并关注图像的光线、亮度和颜色分布，而不被高级内容干扰。

### Goals

1. **Discrimination Ability**: Accurately distinguish between backlit images, enhancement results, and well-lit images.
2. **Detail Awareness**: Better capture fine-grained differences in brightness across different regions of the image.

### Main Methods

1. **Prompt Optimization**:
    - Address the shortcomings of initial prompts by optimizing positive and negative prompts.
    - Utilize a margin ranking loss function $\mathcal{L}_{\text{prompt1}}$ to adjust the similarity between images and prompts, making the prompts more accurately reflect the distribution of lighting and color.
2. **Enhanced Loss Improvement**:
    - Introduce the previous enhancement result $I_{t-1}$ into the loss function $\mathcal{L}_{\text{prompt2}}$ as a constraint.
    - Define multiple margins $m_0, m_1, m_2$ to ensure the enhancement result is closer to a well-lit image while maintaining similarity to the previous enhancement result.
3. **Alternating Iteration**:
    - Adjust the enhancement network after optimizing the prompts.
    - Use the enhancement results to further optimize the prompts, iteratively improving performance.

**Core Idea**: By optimizing the interplay between prompts and the enhancement network, the model can focus on the distribution of lighting, brightness, and color in the image without being distracted by high-level content.