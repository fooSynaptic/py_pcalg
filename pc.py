"""
PC 算法实现模块 (PC Algorithm Implementation)

本模块实现了 PC (Peter-Clark) 算法，用于从观测数据中发现因果结构。
PC 算法是因果发现领域最经典和广泛使用的算法之一。

算法流程:
    1. 初始化完全无向图（所有变量两两相连）
    2. 骨架发现（Skeleton Discovery）：
       - 遍历所有边 (X, Y)
       - 对于每条边，搜索条件集 S 使得 X ⊥ Y | S
       - 如果找到这样的 S，则删除边 X-Y
    3. 方向确定（Edge Orientation）：
       - 识别 V-structures (碰撞器)
       - 应用 Meek 规则传播方向

参考文献:
    Spirtes, P., Glymour, C., & Scheines, R. (2000). 
    Causation, Prediction, and Search (2nd ed.). MIT Press.

作者: py_pcalg contributors
"""

import sys
import numpy as np
import random
from math import sqrt, log
from typing import List, Optional, Tuple, Dict, Union, Callable, Any
import pandas as pd

from utils import (
    Matrix, 
    get_next_set, 
    getNextSet,  # 向后兼容
    independence_test, 
    indTest,  # 向后兼容
    partial_correlation,
    z_statistic,
    pseudoinverse
)
from graph import visualize_graph, grapher


def skeleton(
    suff_stat: List,
    indep_test: Callable,
    alpha: float,
    labels: List,
    fixed_gaps: Optional[np.ndarray] = None,
    fixed_edges: Optional[np.ndarray] = None,
    na_delete: bool = True,
    m_max: float = float('Inf'),
    u2pd: Tuple[str, ...] = ("relaxed", "rand", "retry"),
    solve_confl: bool = False,
    num_cores: int = 1,
    verbose: bool = False
) -> Matrix:
    """
    PC 算法的骨架发现阶段
    
    从完全图开始，通过条件独立性检验逐步删除边，
    得到因果图的骨架结构（无向图）。
    
    Args:
        suff_stat: 充分统计量，通常是 [相关矩阵, 样本数量]
        indep_test: 条件独立性检验函数，签名为 f(x, y, S, suff_stat) -> p_value
        alpha: 显著性水平，p > alpha 时认为条件独立
        labels: 变量标签列表
        fixed_gaps: 固定不存在的边（可选），矩阵形式
        fixed_edges: 固定存在的边（可选），矩阵形式
        na_delete: 是否在 p 值缺失时删除边
        m_max: 最大条件集大小
        u2pd: 处理无向边的策略 ("relaxed", "rand", "retry")
        solve_confl: 是否解决冲突
        num_cores: 并行计算核心数（当前未实现）
        verbose: 是否输出详细信息
        
    Returns:
        Matrix: 邻接矩阵，表示发现的骨架结构
        
    Raises:
        Exception: 如果未指定 labels 参数
        
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # 准备数据
        >>> data = pd.read_csv('data.csv')
        >>> corr = data.corr()
        >>> n = len(data)
        >>> labels = list(range(data.shape[1]))
        >>> # 运行骨架发现
        >>> skel = skeleton([corr, n], indTest, alpha=0.05, labels=labels)
        >>> print(skel.M)  # 打印邻接矩阵
        
    Note:
        这是 PC 算法的核心步骤。算法复杂度取决于图的稀疏程度和
        最大邻居数量。在稀疏图上效率很高。
    """
    # 验证输入
    try:
        p = len(labels)
    except TypeError:
        raise Exception("必须指定 'labels' 参数！")
    
    seq_p = list(range(p))  # 变量索引序列
    
    # 初始化邻接矩阵
    if fixed_gaps is None:
        # 初始化为完全图（所有变量两两相连）
        G = np.ones(shape=(p, p))
    else:
        # 验证固定间隙矩阵
        if fixed_gaps.shape != (p, p):
            raise Exception("fixed_gaps 的维度与数据集不匹配")
        if not np.allclose(fixed_gaps, fixed_gaps.T):
            raise Exception("fixed_gaps 必须是对称矩阵")
        G = np.zeros(shape=(p, p))
    
    # 包装为 Matrix 对象
    G = Matrix(G)
    G.diag(0)  # 对角线设为 0（节点不与自身相连）
    
    # 初始化固定边矩阵
    if fixed_edges is None:
        fixed_edges = np.zeros(shape=(p, p))
    
    # 初始化分离集和最大 p 值矩阵
    sepset = [[None] * p for _ in range(p)]  # 分离集
    pMax = np.full((p, p), float('Inf'))     # 最大 p 值
    pMax = Matrix(pMax)
    pMax.diag(1)
    
    # 主循环变量
    done = False
    order = 0  # 当前条件集大小
    n_edge_tests = [0] * p  # 边检验次数统计
    
    # 主循环：逐步增加条件集大小
    while not done and G.any() and order <= m_max:
        order_plus_one = order + 1
        n_edge_tests[order_plus_one] = 0
        done = True
        
        # 获取当前所有边的索引
        edge_indices = G.which(1)
        num_edges = G.shape[1]
        
        # 遍历所有边
        for i in range(num_edges):
            if i >= len(edge_indices):
                break
                
            x = edge_indices[i, 0]
            y = edge_indices[i, 1]
            
            # 检查边是否存在且不是固定边
            if G.M[y, x] and not fixed_edges[y, x]:
                # 获取 x 的邻居（不包括 y）
                neighbors_bool = G.M[:, x].copy()
                neighbors_bool[y] = 0
                neighbors = [
                    idx for idx in seq_p 
                    if seq_p[idx] and neighbors_bool[idx]
                ]
                num_neighbors = len(neighbors)
                
                # 只有邻居数量足够才能形成条件集
                if num_neighbors >= order:
                    if num_neighbors > order:
                        done = False  # 还需要继续更高阶的检验
                    
                    # 生成初始条件集
                    S = list(range(order + 1))
                    if len(S) == 0:
                        return G
                    
                    # 遍历所有大小为 order 的条件集
                    while True:
                        n_edge_tests[order_plus_one] += 1
                        
                        # 执行独立性检验
                        try:
                            condition_set = [
                                neighbors[s] for s in S 
                                if s < len(neighbors)
                            ]
                            p_value = indep_test(x, y, condition_set, suff_stat)
                        except Exception as e:
                            print(f"检验出错: S={S}, neighbors={neighbors}, error={e}")
                            p_value = None
                        
                        # 处理缺失 p 值
                        if p_value is None:
                            p_value = int(na_delete)
                        
                        # 更新最大 p 值
                        if pMax.M[x, y] < p_value:
                            pMax.M[x, y] = p_value
                        
                        # 如果 p 值大于阈值，删除边并记录分离集
                        if p_value >= alpha:
                            G.M[x, y] = G.M[y, x] = 0
                            try:
                                sepset[x][y] = [neighbors[s] for s in S]
                            except Exception:
                                return G
                            break
                        
                        # 获取下一个条件集
                        next_set = get_next_set(num_neighbors, order, S)
                        if next_set is None or next_set['waslast']:
                            break
                        S = next_set['set']
        
        order += 1
    
    # 对称化最大 p 值矩阵
    for i in range(1, p):
        for j in range(i + 1, p):
            pMax.M[i, j] = pMax.M[j, i] = max(pMax.M[i, j], pMax.M[j, i])
    
    return G


# 保持向后兼容的别名
skeletion = skeleton


def pc(
    suff_stat: List,
    indep_test: Callable,
    alpha: float,
    labels: List,
    p: int,
    fixed_gaps: Optional[np.ndarray] = None,
    fixed_edges: Optional[np.ndarray] = None,
    na_delete: bool = True,
    m_max: float = float('Inf'),
    u2pd: Tuple[str, ...] = ("relaxed", "rand", "retry"),
    skel_method: Tuple[str, ...] = ('stable', 'original'),
    solve_confl: bool = False,
    num_cores: int = 1,
    verbose: bool = False
) -> Matrix:
    """
    PC 算法完整实现
    
    包括骨架发现和边方向确定两个阶段。
    
    Args:
        suff_stat: 充分统计量
        indep_test: 独立性检验函数
        alpha: 显著性水平
        labels: 变量标签
        p: 变量数量
        fixed_gaps: 固定间隙矩阵
        fixed_edges: 固定边矩阵
        na_delete: 缺失值处理
        m_max: 最大条件集大小
        u2pd: 无向边处理策略
        skel_method: 骨架发现方法
        solve_confl: 是否解决冲突
        num_cores: 并行核心数
        verbose: 详细输出
        
    Returns:
        Matrix: 最终的因果图邻接矩阵
        
    Note:
        当前版本仅实现了骨架发现，边方向确定待实现。
    """
    try:
        _ = labels
        _ = p
    except Exception:
        raise Exception("必须指定 'labels' 和 'p' 参数！")
    
    # TODO: 实现完整的 PC 算法，包括方向确定
    # 当前仅返回骨架
    return skeleton(
        suff_stat=suff_stat,
        indep_test=indep_test,
        alpha=alpha,
        labels=labels,
        fixed_gaps=fixed_gaps,
        fixed_edges=fixed_edges,
        na_delete=na_delete,
        m_max=m_max,
        u2pd=u2pd,
        solve_confl=solve_confl,
        num_cores=num_cores,
        verbose=verbose
    )


def demo_pc_algorithm():
    """
    PC 算法演示函数
    
    使用测试数据演示 PC 算法的完整流程，包括：
    1. 数据加载和预处理
    2. 计算相关矩阵和偏相关系数
    3. 执行骨架发现算法
    4. 可视化结果
    
    Example:
        >>> demo_pc_algorithm()
    """
    print("=" * 60)
    print("PC 算法演示")
    print("=" * 60)
    
    # 1. 加载测试数据
    print("\n[1] 加载测试数据...")
    data_df = pd.read_csv('./test_data.csv')
    data = np.array(data_df.iloc[:, :])[:, 1:]  # 跳过索引列
    
    # 2. 计算相关矩阵
    print("[2] 计算相关矩阵...")
    corr_matrix = pd.DataFrame(data).corr()
    
    # 3. 设置变量标签
    # 中英文对照: space, middle(中), Sports(体育讯), min(分), sina(新浪), 
    # match(比赛), player(球员), team(球队), day(日), month(月), 
    # peking(北京), time(时间)
    labels = [
        'space', 'middle', 'Sports', 'min', 'sina', 
        'match', 'player', 'team', 'day', 'month', 
        'peking', 'time'
    ]
    
    # 验证数据维度
    assert data.shape[1] == len(labels), \
        f"数据列数 ({data.shape[1]}) 与标签数 ({len(labels)}) 不匹配"
    
    # 创建标签字典
    label_dict = dict(zip(range(data.shape[1]), labels))
    print(f"   变量数量: {len(labels)}")
    print(f"   样本数量: {data.shape[0]}")
    
    # 4. 演示基础统计计算
    print("\n[3] 演示基础统计计算...")
    
    # 伪逆计算
    rev = pseudoinverse(corr_matrix)
    print(f"   伪逆矩阵形状: {rev.shape}")
    
    # 偏相关系数
    pcor = partial_correlation(1, 2, 3, corr_matrix)
    print(f"   偏相关系数 r(1,2|3) = {pcor:.4f}")
    
    # Z 统计量
    zs = z_statistic(1, 2, 3, corr_matrix, corr_matrix.shape[0])
    print(f"   Z 统计量 = {zs:.4f}")
    
    # 独立性检验
    p_val = independence_test(1, 2, 3, [corr_matrix, corr_matrix.shape[0]])
    print(f"   p 值 = {p_val:.4f}")
    
    # 5. 演示组合生成器
    print("\n[4] 演示组合生成器...")
    next_set = get_next_set(5, 2, [1, 2])
    print(f"   getNextSet(5, 2, [1,2]) = {next_set}")
    print(f"   参考结果: {{'set': [1, 3], 'waslast': False}}")
    
    next_set = get_next_set(5, 2, [4, 5])
    print(f"   getNextSet(5, 2, [4,5]) = {next_set}")
    print(f"   参考结果: {{'set': [4, 5], 'waslast': True}}")
    
    # 6. 运行骨架发现算法
    print("\n[5] 运行骨架发现算法...")
    print("   显著性水平 alpha = 0.05")
    
    graph = skeleton(
        suff_stat=[corr_matrix, corr_matrix.shape[0]],
        indep_test=independence_test,
        alpha=0.05,
        labels=list(range(corr_matrix.shape[0])),
        fixed_gaps=None,
        fixed_edges=None,
        na_delete=True,
        m_max=float('Inf'),
        u2pd=("relaxed", "rand", "retry"),
        solve_confl=False,
        num_cores=1,
        verbose=False
    )
    
    print(f"\n[6] 结果邻接矩阵:")
    print(f"   形状: {graph.shape}")
    print(f"   边数: {int(np.sum(graph.M) / 2)}")
    
    # 7. 可视化
    print("\n[7] 可视化因果图...")
    visualize_graph(graph, label_dict)
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


# 保持向后兼容
debug_trivial = demo_pc_algorithm


if __name__ == "__main__":
    demo_pc_algorithm()
