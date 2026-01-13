import jieba
import jieba.analyse
import re
import numpy as np

def split_sentences(text):
    """
    将文本拆分为句子列表
    """
    # 匹配中英文句号、感叹号、问号作为分隔符
    pattern = re.compile(r'[。！？\?\!]+')
    sentences = pattern.split(text)
    # 过滤掉空句子并去除首尾空格
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences

def get_sentence_similarity(s1, s2):
    """
    计算两个句子的相似度（基于词频交集）
    """
    words1 = set(jieba.lcut(s1))
    words2 = set(jieba.lcut(s2))
    
    if not words1 or not words2:
        return 0.0
    
    # 相似度公式：交集长度 / (log(len1) + log(len2))
    common_words = words1.intersection(words2)
    log_len = np.log(len(words1)) + np.log(len(words2))
    
    if log_len <= 0:
        return 0.0
        
    return len(common_words) / log_len

def extract_summary_textrank(text, ratio=0.4):
    """
    修复版：文本摘要核心函数
    :param text: 输入的长文本
    :param ratio: 摘要比例（保留原句的百分比）
    :return: 摘要文本字符串
    """
    # 基础校验：如果文本太短或句子太少
    if not text or len(text) < 50:
        return "文本过短，无法生成摘要"
    
    sentences = split_sentences(text)
    n = len(sentences)
    
    if n <= 2:
        return text  # 句子太少直接返回全文

    # 1. 构建相似度矩阵
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = get_sentence_similarity(sentences[i], sentences[j])

    # 2. 简单的 PageRank 迭代计算权重
    # 初始分数
    scores = np.ones(n)
    d = 0.85  # 阻尼系数
    for _ in range(20):  # 迭代20次
        prev_scores = np.copy(scores)
        for i in range(n):
            sum_val = 0
            for j in range(n):
                if i != j and similarity_matrix[j][i] > 0:
                    # 计算权重：入链分值 / 出链总权重
                    out_sum = np.sum(similarity_matrix[j])
                    if out_sum > 0:
                        sum_val += (similarity_matrix[j][i] / out_sum) * prev_scores[j]
            scores[i] = (1 - d) + d * sum_val

    # 3. 根据 ratio 计算需要保留的句子数量
    top_n = max(1, int(n * ratio))
    
    # 4. 选取分数最高的句子索引，并按原始顺序排列
    top_indices = np.argsort(scores)[-top_n:]
    top_indices.sort()  # 恢复原有的叙述逻辑顺序
    
    summary_list = [sentences[i] for i in top_indices]
    
    # 5. 拼接成最终摘要，补上句号
    summary = "。".join(summary_list)
    if not summary.endswith(("。", "！", "？")):
        summary += "。"
        
    return summary