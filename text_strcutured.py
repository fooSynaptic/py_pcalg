"""
文本结构化和预处理模块

本模块提供中文文本预处理功能，包括：
- 中文分词 (使用 jieba)
- 停用词过滤
- TF-IDF 特征提取
- 数据向量化

依赖:
    - jieba: 中文分词
    - gensim: 主题建模和文档相似性
    - numpy: 数值计算
    - pandas: 数据处理

Example:
    >>> from text_strcutured import preprocess_data
    >>> tfidf, n_token, texts, dictionary = preprocess_data(
    ...     'corpus.txt', 'stopwords.txt', slic=1000)
"""

import codecs
import sys
import logging
from gensim import corpora, models
import jieba
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO
)


def preprocess_data(
    text_file: str, 
    stopwords_file: str, 
    slice_size: int
) -> Tuple[models.TfidfModel, int, List[List[str]], corpora.Dictionary]:
    """
    预处理文本数据
    
    对原始文本进行分词、停用词过滤、低频词过滤，并构建 TF-IDF 模型。
    
    Args:
        text_file: 原始语料文件路径
        stopwords_file: 停用词文件路径
        slice_size: 处理的文档数量，':' 表示全部
        
    Returns:
        Tuple 包含:
        - tfidf: 训练好的 TF-IDF 模型
        - n_token: 词汇表大小
        - texts: 分词后的文本列表
        - dictionary: Gensim 词典对象
        
    Example:
        >>> tfidf, n_token, texts, dictionary = preprocess_data(
        ...     'corpus.txt', 'stopwords.txt', 500)
    """
    # 读取文档
    with open(text_file, encoding='utf-8') as f:
        documents = f.readlines()
    
    # 提取文本内容并切片
    if slice_size == ':':
        documents = [doc.strip().split('\t')[1] for doc in documents]
    else:
        documents = [doc.strip().split('\t')[1] for doc in documents][:slice_size]
    
    # 加载停用词
    with open(stopwords_file, encoding='utf-8') as f:
        stoplist = [line.strip() for line in f.readlines()]
    
    # 分词并过滤停用词
    texts = [
        [word for word in jieba.cut(document) if word not in stoplist]
        for document in documents
    ]
    
    # 统计词频并过滤低频词
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]
    
    # 构建词典
    dictionary = corpora.Dictionary(texts)
    n_token = len(dictionary.token2id)
    print(f'Total tokens: {n_token}')
    
    # 创建语料库
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # 训练 TF-IDF 模型
    tfidf = models.TfidfModel(corpus)
    
    return tfidf, n_token, texts, dictionary


def vectorize_text(
    tfidf_model: models.TfidfModel,
    texts: List[List[str]],
    dictionary: corpora.Dictionary,
    n_tokens: int
) -> pd.DataFrame:
    """
    将文本转换为向量矩阵
    
    Args:
        tfidf_model: TF-IDF 模型
        texts: 分词后的文本列表
        dictionary: Gensim 词典对象
        n_tokens: 词汇表大小
        
    Returns:
        pd.DataFrame: 向量化后的数据
    """
    # 初始化矩阵
    matrix_shape = (tfidf_model.num_docs, n_tokens)
    real_matrix = np.zeros(shape=matrix_shape)
    
    # 填充 TF-IDF 向量
    for i in range(len(texts)):
        real_vec = real_matrix[i, :]
        vec = dictionary.doc2bow(texts[i])
        vector = tfidf_model[vec]
        for idx, val in vector:
            real_vec[idx] = val
    
    print(f'Saving to CSV format... with shape {real_matrix.shape}')
    
    # 创建表头
    header = np.array(list(dictionary.token2id.keys())).reshape(1, -1)
    print(f"Header shape: {header.shape}")
    
    # 合并表头和数据
    final_array = np.vstack((header, real_matrix))
    data = pd.DataFrame(final_array)
    
    return data


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description='Text preprocessing and vectorization tool'
    )
    parser.add_argument(
        "--f_text", 
        help="Raw corpus file path"
    )
    parser.add_argument(
        "--stop_text", 
        help="Stopwords file path"
    )
    parser.add_argument(
        "--slic", 
        help="Number of documents to process (use ':' for all)"
    )
    
    args = parser.parse_args()
    
    # 设置默认路径
    if not args.f_text or not args.stop_text or not args.slic:
        doc_path = '/Users/ajmd/Downloads/cnews/cnews.test.txt'
        stopwords_path = '/Users/ajmd/data/stopwords/CNstopwords.txt'
        slice_size = 500
    else:
        doc_path = args.f_text
        stopwords_path = args.stop_text
        slice_size = args.slic if args.slic == ':' else int(args.slic)
    
    # 预处理数据
    tfidf, n_token, texts, dictionary = preprocess_data(
        doc_path, stopwords_path, slice_size
    )
    
    # 向量化
    data = vectorize_text(tfidf, texts, dictionary, n_token)
    
    return data


if __name__ == "__main__":
    main()
