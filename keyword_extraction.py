"""
关键词提取模块
支持多种算法：TF-IDF、TextRank
"""
import jieba
import jieba.analyse
from collections import Counter
import re

def extract_keywords_tfidf(text, topK=5):
    """
    使用TF-IDF算法提取关键词
    :param text: 输入文本
    :param topK: 返回关键词数量
    :return: 关键词列表 [(word, weight), ...]
    """
    try:
        if not text or len(text.strip()) < 2:
            return []
        
        # 使用jieba的TF-IDF提取
        keywords = jieba.analyse.extract_tags(
            text, 
            topK=topK, 
            withWeight=True,
            allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'a')  # 限制词性
        )
        
        return keywords
    except Exception as e:
        print(f"TF-IDF关键词提取失败：{str(e)}")
        return []

def extract_keywords_textrank(text, topK=5):
    """
    使用TextRank算法提取关键词
    :param text: 输入文本
    :param topK: 返回关键词数量
    :return: 关键词列表 [(word, weight), ...]
    """
    try:
        if not text or len(text.strip()) < 2:
            return []
        
        # 使用jieba的TextRank提取
        keywords = jieba.analyse.textrank(
            text, 
            topK=topK, 
            withWeight=True,
            allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'a')
        )
        
        return keywords
    except Exception as e:
        print(f"TextRank关键词提取失败：{str(e)}")
        return []

def extract_keywords_hybrid(text, topK=5):
    """
    混合算法：结合TF-IDF和TextRank的结果
    :param text: 输入文本
    :param topK: 返回关键词数量
    :return: 关键词列表 [(word, weight), ...]
    """
    try:
        # 分别提取关键词
        tfidf_keywords = dict(extract_keywords_tfidf(text, topK=topK*2))
        textrank_keywords = dict(extract_keywords_textrank(text, topK=topK*2))
        
        # 合并结果，取平均权重
        all_keywords = set(list(tfidf_keywords.keys()) + list(textrank_keywords.keys()))
        hybrid_results = []
        
        for word in all_keywords:
            tfidf_weight = tfidf_keywords.get(word, 0)
            textrank_weight = textrank_keywords.get(word, 0)
            # 加权平均（TF-IDF权重更高）
            avg_weight = tfidf_weight * 0.6 + textrank_weight * 0.4
            hybrid_results.append((word, avg_weight))
        
        # 按权重排序
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        return hybrid_results[:topK]
    except Exception as e:
        print(f"混合关键词提取失败：{str(e)}")
        return []

def format_keywords_result(keywords, algorithm="混合算法"):
    """
    格式化关键词提取结果为HTML
    :param keywords: 关键词列表 [(word, weight), ...]
    :param algorithm: 使用的算法名称
    :return: HTML格式的结果
    """
    if not keywords:
        return '<div style="color:#ff6600;">⚠️ 未提取到关键词</div>'
    
    html = f'<div class="keyword-section">'
    html += f'<div class="keyword-title">🔑 关键词提取（{algorithm}）</div>'
    html += '<div class="keyword-list">'
    
    for idx, (word, weight) in enumerate(keywords, 1):
        # 根据权重设置颜色深度
        opacity = max(0.5, min(1.0, weight * 2))
        html += f'<span class="keyword-tag" style="opacity:{opacity}">'
        html += f'{idx}. {word} <small>({weight:.3f})</small>'
        html += '</span>'
    
    html += '</div></div>'
    return html

# 初始化jieba（预加载词典）
def init_jieba():
    """初始化jieba分词器"""
    try:
        # 预加载
        jieba.initialize()
        print("✓ jieba分词器初始化成功")
    except Exception as e:
        print(f"✗ jieba分词器初始化失败：{str(e)}")

if __name__ == "__main__":
    # 测试代码
    test_text = """
    人工智能技术在近年来取得了突飞猛进的发展，深度学习、自然语言处理等领域
    的突破性进展，让机器能够更好地理解和生成人类语言。大语言模型的出现，
    更是推动了AI应用的普及和落地。
    """
    
    print("=== TF-IDF算法 ===")
    keywords = extract_keywords_tfidf(test_text)
    for word, weight in keywords:
        print(f"{word}: {weight:.4f}")
    
    print("\n=== TextRank算法 ===")
    keywords = extract_keywords_textrank(test_text)
    for word, weight in keywords:
        print(f"{word}: {weight:.4f}")
    
    print("\n=== 混合算法 ===")
    keywords = extract_keywords_hybrid(test_text)
    for word, weight in keywords:
        print(f"{word}: {weight:.4f}")