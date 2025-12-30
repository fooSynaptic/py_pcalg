"""
图可视化模块 (Graph Visualization Module)

本模块提供因果图的可视化功能，基于 NetworkX 和 Matplotlib。

功能:
    - 将邻接矩阵转换为 NetworkX 图对象
    - 可视化因果图结构
    - 支持自定义节点标签

依赖:
    - networkx: 图数据结构和算法
    - matplotlib: 绑图库

Example:
    >>> from graph import visualize_graph
    >>> from utils import Matrix
    >>> import numpy as np
    >>> # 创建示例邻接矩阵
    >>> adj = Matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    >>> labels = {0: 'A', 1: 'B', 2: 'C'}
    >>> visualize_graph(adj, labels)
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple, Any
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def matrix_to_edge_list(
    adj_matrix, 
    label_dict: Dict[int, str]
) -> List[Tuple[str, str, float]]:
    """
    将邻接矩阵转换为带权重的边列表
    
    Args:
        adj_matrix: Matrix 对象，包含邻接矩阵
        label_dict: 节点索引到标签的映射字典
        
    Returns:
        List[Tuple[str, str, float]]: 边列表，每个元素为 (源节点, 目标节点, 权重)
        
    Example:
        >>> adj = Matrix(np.array([[0, 1], [1, 0]]))
        >>> labels = {0: 'A', 1: 'B'}
        >>> edges = matrix_to_edge_list(adj, labels)
        >>> print(edges)
        [('A', 'B', 1.0)]
    """
    edge_list = []
    rows, cols = adj_matrix.shape
    
    # 遍历上三角矩阵（无向图只需记录一次）
    for i in range(rows):
        for j in range(i + 1, cols):  # 只遍历上三角
            weight = adj_matrix.M[i, j]
            if weight > 0:
                source = label_dict.get(i, str(i))
                target = label_dict.get(j, str(j))
                edge_list.append((source, target, weight))
    
    return edge_list


def visualize_graph(
    adj_matrix,
    label_dict: Dict[int, str],
    title: str = "Causal Graph Skeleton",
    figsize: Tuple[int, int] = (12, 8),
    node_color: str = '#87CEEB',
    edge_color: str = '#4A4A4A',
    node_size: int = 2000,
    font_size: int = 10,
    font_weight: str = 'bold',
    layout: str = 'spring',
    show_edge_weights: bool = False,
    save_path: Optional[str] = None,
    **kwargs
) -> nx.Graph:
    """
    可视化因果图骨架
    
    将邻接矩阵表示的图结构转换为可视化图形。
    
    Args:
        adj_matrix: Matrix 对象，包含邻接矩阵
        label_dict: 节点索引到标签的映射字典
        title: 图标题
        figsize: 图形大小 (宽, 高)
        node_color: 节点颜色
        edge_color: 边颜色
        node_size: 节点大小
        font_size: 字体大小
        font_weight: 字体粗细
        layout: 布局算法 ('spring', 'circular', 'random', 'shell', 'kamada_kawai')
        show_edge_weights: 是否显示边权重
        save_path: 保存图片的路径，None 表示不保存
        **kwargs: 其他传递给 nx.draw 的参数
        
    Returns:
        nx.Graph: NetworkX 图对象
        
    Example:
        >>> from utils import Matrix
        >>> import numpy as np
        >>> adj = Matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]))
        >>> labels = {0: 'X', 1: 'Y', 2: 'Z'}
        >>> G = visualize_graph(adj, labels, title="My Causal Graph")
    """
    # 创建无向图
    G = nx.Graph()
    
    # 转换为边列表并添加到图中
    edge_list = matrix_to_edge_list(adj_matrix, label_dict)
    
    if not edge_list:
        print("警告: 图中没有边")
    
    G.add_weighted_edges_from(edge_list)
    
    # 打印边信息
    print(f"\n图结构信息:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  边列表: {edge_list}")
    
    # 打印低权重边（如果存在）
    low_weight_edges = []
    for node, neighbors in G.adj.items():
        for neighbor, attrs in neighbors.items():
            weight = attrs.get('weight', 1.0)
            if weight < 0.5:
                low_weight_edges.append((node, neighbor, weight))
    
    if low_weight_edges:
        print(f"\n低权重边 (weight < 0.5):")
        for src, tgt, wt in low_weight_edges:
            print(f"  ({src}, {tgt}, {wt:.3f})")
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 选择布局算法
    layout_functions = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'random': nx.random_layout,
        'shell': nx.shell_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'spectral': nx.spectral_layout
    }
    
    layout_func = layout_functions.get(layout, nx.spring_layout)
    pos = layout_func(G)
    
    # 绘制图形
    nx.draw(
        G, 
        pos,
        with_labels=True,
        node_color=node_color,
        edge_color=edge_color,
        node_size=node_size,
        font_size=font_size,
        font_weight=font_weight,
        **kwargs
    )
    
    # 显示边权重（如果需要）
    if show_edge_weights:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图片已保存至: {save_path}")
    
    plt.show()
    
    return G


# 保持向后兼容的别名
grapher = visualize_graph


def create_directed_graph(
    adj_matrix,
    label_dict: Dict[int, str],
    **kwargs
) -> nx.DiGraph:
    """
    创建有向图
    
    将邻接矩阵转换为有向图对象，用于表示因果方向。
    
    Args:
        adj_matrix: Matrix 对象，包含邻接矩阵
        label_dict: 节点索引到标签的映射字典
        **kwargs: 传递给 visualize_graph 的其他参数
        
    Returns:
        nx.DiGraph: NetworkX 有向图对象
        
    Note:
        有向图中 adj_matrix[i, j] = 1 表示存在边 i -> j
    """
    DG = nx.DiGraph()
    
    rows, cols = adj_matrix.shape
    for i in range(rows):
        for j in range(cols):
            if i != j and adj_matrix.M[i, j] > 0:
                source = label_dict.get(i, str(i))
                target = label_dict.get(j, str(j))
                DG.add_edge(source, target, weight=adj_matrix.M[i, j])
    
    return DG


def visualize_directed_graph(
    adj_matrix,
    label_dict: Dict[int, str],
    title: str = "Directed Causal Graph",
    **kwargs
) -> nx.DiGraph:
    """
    可视化有向因果图
    
    Args:
        adj_matrix: Matrix 对象
        label_dict: 节点标签字典
        title: 图标题
        **kwargs: 其他可视化参数
        
    Returns:
        nx.DiGraph: 有向图对象
    """
    DG = create_directed_graph(adj_matrix, label_dict)
    
    plt.figure(figsize=kwargs.get('figsize', (12, 8)))
    pos = nx.spring_layout(DG)
    
    nx.draw(
        DG,
        pos,
        with_labels=True,
        node_color=kwargs.get('node_color', '#FFB6C1'),
        edge_color=kwargs.get('edge_color', '#4A4A4A'),
        node_size=kwargs.get('node_size', 2000),
        font_size=kwargs.get('font_size', 10),
        font_weight='bold',
        arrows=True,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1'
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return DG


# 示例用法和测试
if __name__ == "__main__":
    from utils import Matrix
    
    # 创建示例邻接矩阵
    test_adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    
    test_matrix = Matrix(test_adj)
    test_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    print("测试图可视化模块")
    print("=" * 40)
    
    # 测试无向图可视化
    G = visualize_graph(
        test_matrix,
        test_labels,
        title="测试无向图",
        layout='circular'
    )
