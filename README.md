# py_pcalg

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**py_pcalg** 是一个纯 Python 实现的 **PC 算法**（Peter-Clark Algorithm），用于从观测数据中发现因果结构。

## <img src=".github/icons/book.svg" width="16" height="16" alt="book"> 简介

PC 算法是因果发现（Causal Discovery）领域最经典和广泛使用的算法之一，由 Peter Spirtes 和 Clark Glymour 于 1991 年提出。该算法基于条件独立性检验，从数据中学习因果图的结构。

本项目旨在填补 Python 生态系统中轻量级 PC 算法实现的空白，提供一个易于理解和使用的因果发现工具。

### 主要特性

- <img src=".github/icons/microscope.svg" width="16" height="16" alt="microscope"> **骨架发现**：通过条件独立性检验发现变量间的关联结构
- <img src=".github/icons/chart.svg" width="16" height="16" alt="chart"> **可视化支持**：内置基于 NetworkX 和 Matplotlib 的图可视化功能
- <img src=".github/icons/note.svg" width="16" height="16" alt="note"> **清晰的代码结构**：详细的注释和类型提示
- <img src=".github/icons/test_tube.svg" width="16" height="16" alt="test_tube"> **即开即用**：提供测试数据和演示脚本

## <img src=".github/icons/rocket.svg" width="16" height="16" alt="rocket"> 快速开始

### 安装依赖

```bash
pip install numpy pandas matplotlib networkx
```

### 基本使用

```python
import pandas as pd
from pc import skeleton, demo_pc_algorithm
from utils import independence_test

# 方法 1: 运行演示
demo_pc_algorithm()

# 方法 2: 使用自己的数据
data = pd.read_csv('your_data.csv')
corr = data.corr()
n_samples = len(data)
labels = list(range(data.shape[1]))

# 运行骨架发现
graph = skeleton(
    suff_stat=[corr, n_samples],
    indep_test=independence_test,
    alpha=0.05,
    labels=labels
)

# 打印邻接矩阵
print(graph.M)
```

### 可视化结果

```python
from graph import visualize_graph

# 创建标签字典
label_dict = {i: f'Var_{i}' for i in range(graph.shape[0])}

# 可视化
visualize_graph(graph, label_dict, title="发现的因果骨架")
```

## <img src=".github/icons/book.svg" width="16" height="16" alt="book"> 算法原理

### PC 算法流程

1. **初始化**：创建完全无向图（所有变量两两相连）

2. **骨架发现**（Skeleton Discovery）：
   - 对于每条边 (X, Y)，搜索条件集 S
   - 执行条件独立性检验：X ⊥ Y | S
   - 如果 p-value > α，则删除边 X-Y，并记录分离集 S

3. **方向确定**（Edge Orientation）：
   - 识别 V-structures（碰撞器）：X → Z ← Y
   - 应用 Meek 规则传播方向

### 条件独立性检验

本实现使用 **Fisher's Z 变换** 进行条件独立性检验：

```
Z = √(n - |S| - 3) × 0.5 × log((1+r)/(1-r))
```

其中：
- `n` 是样本数量
- `|S|` 是条件集大小
- `r` 是偏相关系数

## <img src=".github/icons/folder.svg" width="16" height="16" alt="folder"> 项目结构

```
py_pcalg/
├── pc.py           # PC 算法主模块
├── utils.py        # 工具函数（Matrix 类、统计检验等）
├── graph.py        # 图可视化模块
├── test_data.csv   # 测试数据
├── demo/           # 演示文件夹
│   ├── demo.fi
│   └── WechatIMG7.png
└── README.md       # 本文档
```

## <img src=".github/icons/chart.svg" width="16" height="16" alt="chart"> API 文档

### `skeleton()`

骨架发现函数，PC 算法的核心。

```python
skeleton(
    suff_stat,      # 充分统计量 [相关矩阵, 样本数]
    indep_test,     # 独立性检验函数
    alpha,          # 显著性水平 (如 0.05)
    labels,         # 变量标签列表
    fixed_gaps=None,    # 固定不存在的边
    fixed_edges=None,   # 固定存在的边
    m_max=inf,          # 最大条件集大小
    verbose=False       # 详细输出
) -> Matrix
```

### `Matrix` 类

邻接矩阵包装类。

```python
from utils import Matrix
import numpy as np

m = Matrix(np.ones((5, 5)))
m.diag(0)           # 对角线设为 0
m.any()             # 检查是否存在边
m.which(1)          # 获取所有值为 1 的位置
```

### `visualize_graph()`

图可视化函数。

```python
visualize_graph(
    adj_matrix,     # Matrix 对象
    label_dict,     # 标签字典 {索引: 标签}
    title="...",    # 图标题
    layout='spring', # 布局算法
    save_path=None  # 保存路径
) -> nx.Graph
```

## <img src=".github/icons/link.svg" width="16" height="16" alt="link"> 相关工作

如果您对因果发现有更多需求，推荐关注以下项目和资源：

### Python 库

| 项目 | 描述 | 链接 |
|------|------|------|
| **causal-learn** | CMU 开发的因果发现工具包，实现了 PC、FCI、GES 等多种算法 | [GitHub](https://github.com/cmu-phil/causal-learn) |
| **gCastle** | 华为开发的因果结构学习工具链 | [GitHub](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle) |
| **DoWhy** | Microsoft 的因果推理库，侧重因果效应估计 | [GitHub](https://github.com/py-why/dowhy) |
| **pgmpy** | 概率图模型库，包含结构学习功能 | [GitHub](https://github.com/pgmpy/pgmpy) |
| **Tigramite** | 时间序列因果发现，支持 PCMCI 算法 | [GitHub](https://github.com/jakobrunge/tigramite) |
| **CausalNex** | McKinsey 开发的贝叶斯网络因果推理库 | [GitHub](https://github.com/quantumblacklabs/causalnex) |

### R 语言

| 项目 | 描述 | 链接 |
|------|------|------|
| **pcalg** | 原始 PC 算法的 R 实现（本项目的参考） | [CRAN](https://CRAN.R-project.org/package=pcalg) |
| **bnlearn** | 贝叶斯网络结构学习和参数学习 | [CRAN](https://cran.r-project.org/package=bnlearn) |

### 学术资源

#### 经典论文

1. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
   - PC 算法的权威著作

2. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
   - 因果推理领域的经典教材

3. Kalisch, M., & Bühlmann, P. (2007). Estimating high-dimensional directed acyclic graphs with the PC-algorithm. *Journal of Machine Learning Research*, 8, 613-636.
   - PC 算法的高维扩展

#### 综述论文

1. Glymour, C., Zhang, K., & Spirtes, P. (2019). Review of causal discovery methods based on graphical models. *Frontiers in Genetics*, 10, 524.

2. Vowels, M. J., Camgoz, N. C., & Mayol-Cuevas, P. (2022). D'ya like DAGs? A survey on structure learning and causal discovery. *ACM Computing Surveys*.

### 在线课程

- [Coursera: A Crash Course in Causality](https://www.coursera.org/learn/crash-course-in-causality) - 宾夕法尼亚大学
- [Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course) - Brady Neal

## <img src=".github/icons/warning.svg" width="16" height="16" alt="warning"> 注意事项

1. **因果假设**：PC 算法依赖以下假设：
   - 因果马尔可夫条件
   - 因果忠实性假设
   - 无潜在混淆变量（可用 FCI 算法处理）

2. **样本量**：条件独立性检验需要足够的样本量，小样本可能导致不可靠结果

3. **计算复杂度**：在密集图上，算法复杂度可能较高

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📮 联系

如有问题，请通过 GitHub Issues 联系。

---

*本项目持续更新中，欢迎 Star <img src=".github/icons/star.svg" width="16" height="16" alt="star"> 和 Fork！*
