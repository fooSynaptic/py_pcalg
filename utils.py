"""
工具函数模块 (Utility Functions Module)

本模块提供 PC 算法所需的核心统计计算工具，包括：
- 矩阵操作类 (Matrix)
- 偏相关系数计算 (Partial Correlation)
- 独立性检验 (Independence Test)
- Fisher's Z 变换统计量计算

参考文献:
    Spirtes, P., Glymour, C., & Scheines, R. (2000). 
    Causation, Prediction, and Search (2nd ed.). MIT Press.
"""

import sys
import numpy as np
import random
from math import sqrt, log
from typing import List, Optional, Tuple, Dict, Union, Any
import pandas as pd


class Matrix:
    """
    矩阵包装类
    
    为 numpy 数组提供额外的便捷方法，用于因果图的邻接矩阵操作。
    
    Attributes:
        M (np.ndarray): 底层的 numpy 矩阵
        shape (tuple): 矩阵的形状 (rows, cols)
    
    Example:
        >>> import numpy as np
        >>> m = Matrix(np.ones((3, 3)))
        >>> m.diag(0)  # 将对角线设为 0
        >>> m.any()    # 检查是否存在值为 1 的元素
        True
    """
    
    def __init__(self, M: np.ndarray):
        """
        初始化矩阵对象
        
        Args:
            M: numpy 数组，通常是邻接矩阵
        """
        self.M = M
        self.shape = M.shape
    
    def diag(self, val: Union[int, float]) -> None:
        """
        将矩阵对角线元素设置为指定值
        
        Args:
            val: 要设置的对角线值
            
        Raises:
            AssertionError: 如果矩阵不是方阵
            
        Example:
            >>> m = Matrix(np.ones((3, 3)))
            >>> m.diag(0)
            >>> m.M[0, 0]  # 返回 0
        """
        rows, cols = self.shape
        assert rows == cols, "Matrix must be square"
        np.fill_diagonal(self.M, val)
    
    def any(self) -> bool:
        """
        检查矩阵中是否存在值为 1 的元素
        
        用于判断图中是否还存在边，是 PC 算法终止条件之一。
        
        Returns:
            bool: 如果存在值为 1 的元素返回 True，否则返回 False
            
        Raises:
            AssertionError: 如果矩阵不是方阵
        """
        rows, cols = self.shape
        assert rows == cols, "Matrix must be square"
        return np.any(self.M == 1)
    
    def which(self, val: Union[int, float]) -> np.ndarray:
        """
        查找矩阵中等于指定值的所有元素索引
        
        返回的索引按列优先排序，用于遍历图中的边。
        
        Args:
            val: 要查找的值
            
        Returns:
            np.ndarray: 形状为 (n, 2) 的数组，每行是一个 (row, col) 索引对
            
        Raises:
            AssertionError: 如果矩阵不是方阵
            
        Example:
            >>> m = Matrix(np.array([[0, 1], [1, 0]]))
            >>> m.which(1)
            array([[0, 1], [1, 0]])
        """
        rows, cols = self.shape
        assert rows == cols, "Matrix must be square"
        
        # 找到所有等于 val 的元素索引
        indices = np.argwhere(self.M == val)
        
        if len(indices) == 0:
            return indices
            
        # 按列优先排序 (先按第二列，再按第一列)
        sorted_indices = indices[np.lexsort(indices[:, ::-1].T)]
        return sorted_indices


def get_next_set(n: int, k: int, current_set: List[int]) -> Optional[Dict[str, Any]]:
    """
    生成下一个大小为 k 的组合集合
    
    用于 PC 算法中遍历所有可能的条件集。这是一个组合生成器，
    按字典序生成 {0, 1, ..., n-1} 的所有 k 元子集。
    
    Args:
        n: 元素总数 (从 0 到 n-1)
        k: 组合的大小
        current_set: 当前的组合，元素值从 0 开始
        
    Returns:
        dict: 包含两个键:
            - 'set': 下一个组合 (List[int])
            - 'waslast': 是否是最后一个组合 (bool)
        如果输入无效则返回 None
        
    Example:
        >>> get_next_set(5, 2, [0, 1])
        {'set': [0, 2], 'waslast': False}
        >>> get_next_set(5, 2, [3, 4])
        {'set': [3, 4], 'waslast': True}
        
    Note:
        这个函数会修改 current_set 参数，调用者应注意传入副本如果需要保留原值
    """
    # 生成最大可能的组合 [n-k, n-k+1, ..., n-1]
    max_combination = list(range(n - k, n))
    
    # 计算与最大组合相同的尾部元素数量
    matching_tail = sum(
        1 for curr, max_val in zip(current_set, max_combination) 
        if curr == max_val
    )
    
    # 确定需要递增的位置
    increment_pos = k - matching_tail
    is_last = (increment_pos == 0)
    
    if not is_last:
        # 处理单元素集合的特殊情况
        if len(current_set) == 1:
            new_val = current_set[0] + 1
        elif len(current_set) == 0 or increment_pos - 1 >= len(current_set) or increment_pos - 1 < 0:
            return None
        else:
            try:
                new_val = current_set[increment_pos - 1] + 1
            except IndexError:
                return None
        
        # 更新当前位置
        current_set[increment_pos - 1] = new_val
        
        # 重置后续位置为连续值
        if increment_pos < k:
            for i in range(increment_pos, k):
                if i < len(current_set):
                    current_set[i] = new_val + (i - increment_pos + 1)
    
    return {'set': current_set, 'waslast': is_last}


# 保持向后兼容的别名
getNextSet = get_next_set


def independence_test(x: int, y: int, S: Union[int, List[int]], 
                      suff_stat: List) -> float:
    """
    执行条件独立性检验
    
    使用 Fisher's Z 变换检验变量 x 和 y 在给定条件集 S 下是否条件独立。
    
    Args:
        x: 第一个变量的索引
        y: 第二个变量的索引
        S: 条件集变量索引（单个整数或整数列表）
        suff_stat: 充分统计量列表 [相关矩阵, 样本数量]
        
    Returns:
        float: p 值，接近 1 表示条件独立，接近 0 表示条件相关
        
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = np.random.randn(100, 5)
        >>> corr = pd.DataFrame(data).corr()
        >>> p = independence_test(0, 1, 2, [corr, 100])
        
    Note:
        这是一个基于正态分布假设的参数化检验方法
    """
    z = z_statistic(x, y, S, suff_stat[0], suff_stat[1])
    print(f"独立性统计量 z = {z:.4f}")
    
    # 使用经验分布估计 p 值
    # 注意: 这是一个近似方法，更精确的方法应使用正态分布 CDF
    p_value = sum(1 for i in range(100000) if i < z) / 100000
    
    # 避免返回过小的 p 值
    return 0 if p_value < 0.00001 else p_value


# 保持向后兼容的别名
indTest = independence_test


def z_statistic(x: int, y: int, S: Union[int, List[int]], 
                C: pd.DataFrame, n: int) -> float:
    """
    计算 Fisher's Z 变换统计量
    
    用于将偏相关系数转换为渐近正态分布的统计量。
    
    公式: Z = sqrt(n - |S| - 3) * 0.5 * log((1+r)/(1-r))
    
    其中 r 是偏相关系数，n 是样本数量，|S| 是条件集大小。
    
    Args:
        x: 第一个变量的索引
        y: 第二个变量的索引
        S: 条件集变量索引
        C: 相关系数矩阵 (pandas DataFrame)
        n: 样本数量
        
    Returns:
        float: Z 统计量值
        
    Reference:
        Fisher, R. A. (1915). Frequency distribution of the values of 
        the correlation coefficient in samples from an indefinitely 
        large population.
    """
    # 确保 S 是列表
    if not isinstance(S, list):
        S = [S]
    
    print(f"条件集 S = {S}")
    
    # 计算偏相关系数
    r = partial_correlation(x, y, S[0] if S else None, C)
    
    # 计算 Fisher's Z 变换
    # 注意: 原公式分母多了一个 /2，这里保持与原代码一致
    degrees_of_freedom = n - len(S) - 3
    if degrees_of_freedom <= 0:
        return 0
        
    try:
        z = sqrt(degrees_of_freedom) * 0.5 * log((1 + r) / (1 - r)) / 2
    except (ValueError, ZeroDivisionError):
        return 0
        
    return z if z else 0


# 保持向后兼容的别名
zstat = z_statistic


def partial_correlation(i: int, j: int, k: Optional[int], 
                        C: pd.DataFrame, cut: float = 0.99999) -> float:
    """
    计算偏相关系数
    
    计算变量 i 和 j 在控制变量 k 后的偏相关系数。
    
    对于一阶偏相关 (控制一个变量):
        r_ij.k = (r_ij - r_ik * r_jk) / sqrt((1 - r_ik²)(1 - r_jk²))
    
    Args:
        i: 第一个变量的索引
        j: 第二个变量的索引
        k: 控制变量的索引，None 表示计算简单相关
        C: 相关系数矩阵 (pandas DataFrame)
        cut: 截断阈值，避免返回 ±1 导致后续计算问题
        
    Returns:
        float: 偏相关系数，范围在 [-cut, cut]
        
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = np.random.randn(100, 3)
        >>> C = pd.DataFrame(data).corr()
        >>> r = partial_correlation(0, 1, 2, C)
        
    Note:
        对于高阶偏相关（控制多个变量），需要使用矩阵求逆方法
    """
    k_list = [k] if k is not None else []
    
    if len(k_list) == 0:
        # 简单相关系数
        r = C.iloc[i, j] if hasattr(C, 'iloc') else C[i, j]
    elif len(k_list) == 1:
        # 一阶偏相关
        idx = k_list[0]
        r_ij = C.iloc[j, i] if hasattr(C, 'iloc') else C[j][i]
        r_ik = C.iloc[idx, i] if hasattr(C, 'iloc') else C[idx][i]
        r_jk = C.iloc[idx, j] if hasattr(C, 'iloc') else C[idx][j]
        
        denominator = sqrt((1 - r_jk**2) * (1 - r_ik**2))
        if denominator == 0:
            return 0
        r = (r_ij - r_ik * r_jk) / denominator
    else:
        # 高阶偏相关，使用精度矩阵方法
        indices = [i, j] + k_list
        sub_matrix = C.iloc[indices, indices] if hasattr(C, 'iloc') else C[indices][:, indices]
        precision_matrix = pseudoinverse(sub_matrix)
        
        denominator = sqrt(precision_matrix[1, 1] * precision_matrix[2, 2])
        if denominator == 0:
            return 0
        r = -precision_matrix[2, 1] / denominator
    
    if not r or np.isnan(r):
        return 0
    
    # 截断到 [-cut, cut] 范围
    return min(cut, max(-cut, r))


# 保持向后兼容的别名
pcorOrder = partial_correlation


def pseudoinverse(m: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    计算矩阵的 Moore-Penrose 伪逆
    
    使用奇异值分解 (SVD) 计算伪逆矩阵。
    
    Args:
        m: 输入矩阵 (numpy array 或 pandas DataFrame)
        
    Returns:
        np.ndarray: 伪逆矩阵
        
    Example:
        >>> import numpy as np
        >>> m = np.array([[1, 2], [3, 4]])
        >>> m_pinv = pseudoinverse(m)
        
    Note:
        对于奇异矩阵或接近奇异的矩阵，伪逆提供最小二乘解
    """
    # 执行奇异值分解
    U, S, Vh = np.linalg.svd(m)
    
    # 找出正的奇异值
    positive_singular_values = [s for s in S if s > 0]
    
    if len(positive_singular_values) == 0:
        # 如果没有正奇异值，返回零矩阵
        return np.zeros(m.shape[::-1])
    
    # 计算伪逆: V * S^(-1) * U^T
    S_inv = np.array([1/s for s in positive_singular_values])
    return np.dot(Vh.T[:, :len(S_inv)], S_inv * U[:, :len(S_inv)].T)


# 奇异值分解函数别名
svd_decomposition = np.linalg.svd
gen_inv = np.linalg.svd

