"""
命名实体识别模块（NER）
识别文本中的人名、地名、机构名等
"""
import jieba
import jieba.posseg as pseg
import re
from collections import defaultdict

def extract_entities_jieba(text):
    """
    使用jieba的词性标注提取命名实体
    :param text: 输入文本
    :return: 实体字典 {'person': [...], 'location': [...], 'organization': [...]}
    """
    try:
        if not text or len(text.strip()) < 2:
            return {'person': [], 'location': [], 'organization': [], 'time': []}
        
        entities = {
            'person': [],      # 人名
            'location': [],    # 地名
            'organization': [], # 机构名
            'time': []         # 时间
        }
        
        # 词性标注
        words = pseg.cut(text)
        
        for word, flag in words:
            # 人名：nr
            if flag == 'nr' and len(word) >= 2:
                entities['person'].append(word)
            # 地名：ns
            elif flag == 'ns' and len(word) >= 2:
                entities['location'].append(word)
            # 机构名：nt
            elif flag == 'nt' and len(word) >= 2:
                entities['organization'].append(word)
            # 时间：t
            elif flag == 't' and len(word) >= 2:
                entities['time'].append(word)
        
        # 去重但保持顺序
        for key in entities:
            seen = set()
            unique_list = []
            for item in entities[key]:
                if item not in seen:
                    seen.add(item)
                    unique_list.append(item)
            entities[key] = unique_list
        
        return entities
        
    except Exception as e:
        print(f"命名实体识别失败：{str(e)}")
        return {'person': [], 'location': [], 'organization': [], 'time': []}

def extract_entities_pattern(text):
    """
    使用正则表达式模式匹配提取实体（辅助方法）
    :param text: 输入文本
    :return: 实体字典
    """
    try:
        entities = {
            'email': [],
            'phone': [],
            'url': [],
            'date': []
        }
        
        # 邮箱
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['email'] = re.findall(email_pattern, text)
        
        # 电话号码（中国）
        phone_pattern = r'1[3-9]\d{9}'
        entities['phone'] = re.findall(phone_pattern, text)
        
        # URL
        url_pattern = r'https?://[^\s]+'
        entities['url'] = re.findall(url_pattern, text)
        
        # 日期（简单匹配）
        date_pattern = r'\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?'
        entities['date'] = re.findall(date_pattern, text)
        
        return entities
        
    except Exception as e:
        print(f"模式匹配提取失败：{str(e)}")
        return {'email': [], 'phone': [], 'url': [], 'date': []}

def extract_all_entities(text):
    """
    综合提取所有实体
    :param text: 输入文本
    :return: 合并后的实体字典
    """
    # 基于词性的实体
    jieba_entities = extract_entities_jieba(text)
    
    # 基于模式的实体
    pattern_entities = extract_entities_pattern(text)
    
    # 合并
    all_entities = {**jieba_entities, **pattern_entities}
    
    return all_entities

def format_entities_result(entities):
    """
    格式化实体识别结果为HTML
    :param entities: 实体字典
    :return: HTML格式的结果
    """
    # 实体类型的中文名称和emoji
    entity_types = {
        'person': ('👤 人名', '#4CAF50'),
        'location': ('📍 地名', '#2196F3'),
        'organization': ('🏢 机构', '#FF9800'),
        'time': ('⏰ 时间', '#9C27B0'),
        'email': ('📧 邮箱', '#00BCD4'),
        'phone': ('📱 电话', '#E91E63'),
        'url': ('🔗 链接', '#607D8B'),
        'date': ('📅 日期', '#795548')
    }
    
    # 统计实体数量
    total_count = sum(len(v) for v in entities.values())
    
    if total_count == 0:
        return '<div style="color:#ff6600;">⚠️ 未识别到命名实体</div>'
    
    html = '<div class="entity-section">'
    html += f'<div class="entity-title">🏷️ 命名实体识别（共{total_count}个）</div>'
    
    for entity_type, entity_list in entities.items():
        if entity_list and entity_type in entity_types:
            type_name, color = entity_types[entity_type]
            html += f'<div class="entity-group">'
            html += f'<div class="entity-group-title" style="color:{color}">{type_name}</div>'
            html += '<div class="entity-list">'
            
            for entity in entity_list:
                html += f'<span class="entity-tag" style="border-color:{color}">{entity}</span>'
            
            html += '</div></div>'
    
    html += '</div>'
    return html

if __name__ == "__main__":
    # 测试代码
    test_text = """
    2024年1月，OpenAI公司在美国旧金山发布了最新的GPT-4模型。
    CEO萨姆·奥特曼表示，这标志着人工智能进入了新的阶段。
    联系方式：contact@openai.com，电话：13800138000。
    更多信息请访问：https://openai.com
    """
    
    print("=== 命名实体识别 ===")
    entities = extract_all_entities(test_text)
    
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"{entity_type}: {entity_list}")