import os
import re
import json
import warnings
import base64
import time
import urllib.parse
import http.client
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from volcenginesdkarkruntime import Ark

# 禁用冗余警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ========== 豆包API配置 ==========
ARK_API_KEY = "a3560ff1-1d8d-49ba-92a7-7c89a9ec18d4" 
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "doubao-seed-1-6-lite-251015" 
ARK_IMAGE_MODEL = "doubao-seedream-4-5-251128" 
ARK_VIDEO_MODEL = "doubao-seedance-1-5-pro-251215" 

# 初始化客户端
client = Ark(base_url=ARK_BASE_URL, api_key=ARK_API_KEY)

# ========== 文件配置 ==========
UPLOAD_FOLDER = '../tmp/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 模型路径
TEXT_CATEGORY_MODEL_PATH = '../tmp/text_category_model.h5'
SENTIMENT_MODEL_PATH = '../tmp/sentiment_model.h5'
SENTIMENT_DICTS_PATH = '../tmp/sentiment_dicts.csv'
VOCAB_DIR = '../data/cnews.vocab.txt'

# 全局模型变量
_sentiment_dicts = None
_sentiment_model = None
_translation_model_loaded = False

# ========== 工具函数 ==========
def allowed_file(filename):
    """精简版：校验文件格式"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_invalid_chars(text):
    """修复版：清理非法字符"""
    if not text:
        return ""
    control_chars = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    return control_chars.sub(' ', text).strip()

def normalize_text(text):
    """修复版：文本格式规范化"""
    if not text:
        return ""
    text = clean_invalid_chars(text)
    try:
        text = json.loads(f'"{text}"')
    except:
        pass
    text = text.replace('\n', '<br>')
    text = text.replace('　', ' ')
    text = re.sub(r'[\u200b\u200c\u200d\r]', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    punct_map = {
        ',': '，', '.': '。', '?': '？', '!': '！',
        ':': '：', ';': '；', '(': '（', ')': '）',
        '[': '【', ']': '】'
    }
    for en, cn in punct_map.items():
        text = text.replace(en, cn)
    
    text = re.sub(r'(\d+)。(\d+)', r'\1.\2', text)
    text = re.sub(r'(\d+\.\d+)。(\d+)', r'\1.\2', text)
    
    return text

def image_to_base64(image_path):
    """修复版：图片转base64"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            ext = image_path.rsplit('.', 1)[1].lower()
            mime_types = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'gif': 'image/gif',
                'webp': 'image/webp'}
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            return f"data:{mime_type};base64,{base64_data}"
    except Exception as e:
        print(f"图片转换失败：{str(e)}")
        return None

def markdown_to_normal(text):
    """新增：Markdown格式转正常显示"""
    if not text: return ""
    
    text = re.sub(r'^#{1,4}\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'^\*\s*(.*?)$', r'• \1', text, flags=re.MULTILINE)
    
    return text

def get_fresh_video_url(task_id):
    """完整版：刷新URL功能"""
    try:
        get_result = client.content_generation.tasks.get(task_id=task_id)
        if get_result.status == "succeeded":
            return get_result.content.video_url
        return None
    except Exception as e:
        print(f"刷新视频URL失败：{e}")
        return None

def format_analysis(results):
    '''修复版：格式化分析结果为HTML（统一显示）'''
    if not results:
        return ""
    
    html = '<div class="analysis-section">'
    html += '<div class="analysis-title">🔍 智能分析</div>'
    
    # 文本分类
    if 'category' in results:
        cat = results['category']
        html += f'<div class="analysis-item">📌 <b>文本分类：</b>{cat["label"]} <span style="color:#4CAF50;">(置信度：{cat["score"]:.2f})</span></div>'
    
    # 情感分析
    if 'sentiment' in results:
        sent = results['sentiment']
        emoji = {'positive': '😊', 'negative': '😢', 'neutral': '😐'}.get(sent['label'], '❤️')
        html += f'<div class="analysis-item">{emoji} <b>情感倾向：</b>{sent["label"]} <span style="color:#4CAF50;">(置信度：{sent["score"]:.2f})</span></div>'
    
    # 关键词提取
    if 'keywords' in results and results['keywords']:
        html += '<div class="analysis-item">🔑 <b>关键词：</b>'
        keywords_list = []
        for word, weight in results['keywords'][:5]:
            keywords_list.append(f'<span class="keyword-inline">{word}</span>')
        html += ' '.join(keywords_list)
        html += '</div>'
    
    # 文本摘要
    if 'summary' in results and results['summary']:
        summary_text = results['summary']
        if len(summary_text) > 100:
            summary_text = summary_text[:100] + '...'
        html += f'<div class="analysis-item">📝 <b>文本摘要：</b>{summary_text}</div>'
    
    # 命名实体识别
    if 'entities' in results and results['entities']:
        entities = results['entities']
        entity_parts = []
        
        if entities.get('person'):
            entity_parts.append(f'👤人名: {", ".join(entities["person"][:3])}')
        if entities.get('location'):
            entity_parts.append(f'📍地名: {", ".join(entities["location"][:3])}')
        if entities.get('organization'):
            entity_parts.append(f'🏢机构: {", ".join(entities["organization"][:3])}')
        if entities.get('time'):
            entity_parts.append(f'⏰时间: {", ".join(entities["time"][:2])}')
        
        if entity_parts:
            html += '<div class="analysis-item">🏷️ <b>实体识别：</b><br>'
            html += '<br>'.join(entity_parts)
            html += '</div>'
    
    html += '</div>'
    return html

# ========== 模型初始化 ==========
def init_models():
    """修复版：初始化所有模型"""
    global _sentiment_dicts, _sentiment_model, _translation_model_loaded
    print("=" * 50)
    print("系统初始化中...")
    print("=" * 50)
    
    try:
        # 1. 检查文本分类模型
        if os.path.exists(TEXT_CATEGORY_MODEL_PATH):
            print("✓ 文本分类模型已就绪")
        else:
            print("✗ 文本分类模型不存在")
        
        # 2. 加载情感分析模型
        if os.path.exists(SENTIMENT_MODEL_PATH):
            from sentiment_analysis import load_sentiment_deps
            _sentiment_dicts, _sentiment_model = load_sentiment_deps(
                SENTIMENT_MODEL_PATH, SENTIMENT_DICTS_PATH)
            if _sentiment_dicts is not None:
                print("✓ 情感分析模型加载成功")
            else:
                print("✗ 情感分析模型加载失败")
        else:
            print("✗ 情感分析模型不存在")
        
        # 3. 加载机器翻译模型
        try:
            from machine_translation import load_translation_model
            load_translation_model()
            _translation_model_loaded = True
            print("✓ 机器翻译模型加载成功")
        except Exception as e:
            _translation_model_loaded = False
            print(f"✗ 机器翻译模型加载失败：{str(e)}")
        
        # 4. 初始化新增的NLP模块
        try:
            from keyword_extraction import init_jieba
            init_jieba()
            print("✓ 关键词提取模块已就绪")
        except Exception as e:
            print(f"✗ 关键词提取模块初始化失败：{str(e)}")
        
        print("=" * 50)
        print("系统初始化完成！")
        print("=" * 50)
    except Exception as e:
        print(f"模型初始化失败：{str(e)}")

# ========== 核心功能函数 ==========
def generate_image(prompt):
    """精简版：生成图片"""
    try:
        resp = client.images.generate(
            model=ARK_IMAGE_MODEL, prompt=prompt, size="2K", response_format="url"
        )
        if resp and resp.data:
            url = resp.data[0].url
            safe_prompt = normalize_text(prompt)
            return f'<b>🎨 图像已生成</b><br><img src="{url}" class="message-image" style="max-width:100%; border-radius:8px; margin-top:10px;"><br><small>提示词：{safe_prompt}</small>'
        return "❌ 图像生成失败"
    except Exception as e:
        return f"❌ 图像生成错误：{str(e)}"

def generate_video(prompt):
    """融合版：生成视频"""
    try:
        create_result = client.content_generation.tasks.create(
            model=ARK_VIDEO_MODEL,
            content=[{"type": "text", "text": f"{prompt} --duration 5 --watermark true"}]
        )
        task_id = create_result.id
        
        for _ in range(100):
            time.sleep(5)
            get_result = client.content_generation.tasks.get(task_id=task_id)
            status = get_result.status
            
            if status == "succeeded":
                video_url = get_result.content.video_url
                if video_url:
                    res_html = (
                        f'<b>🎬 视频生成成功</b><br>'
                        f'<div>分辨率：{getattr(get_result, "resolution", "N/A")} | 时长：{getattr(get_result, "duration", "5")}秒</div>'
                        f'<div style="color:#ff6600; font-size:12px; margin:8px 0;">⚠️ 视频链接有效期24小时，过期可刷新</div>'
                        f'<a href="{video_url}" target="_blank" style="display:inline-block; margin-top:5px; padding:8px 15px; background:#4CAF50; color:white; border-radius:4px; text-decoration:none;">点击打开并下载视频</a>'
                        f'<div style="font-size:12px; margin-top:8px;">任务ID：<code>{task_id}</code>（刷新链接用）</div>'
                    )
                    return res_html
                return "❌ 未获取到视频链接"
            
            elif status == "failed":
                return f"❌ 视频生成失败：{getattr(get_result, 'error', '未知错误')}"
                
        return f"⚠️ 任务超时（最大等待100次），请稍后查询 ID: {task_id}"
    except Exception as e:
        return f"❌ 视频生成启动失败：{str(e)}"

def chat(sentence='', image_paths=None, **switches):
    """融合版：核心对话功能（修复版：正确收集和显示分析结果）"""
    try:
        # 功能分发（生成图片/视频）
        if switches.get('enable_image_gen') and "生成图片" in sentence:
            prompt = re.sub(r'生成图片[:：]?', '', sentence).strip()
            return generate_image(prompt)
            
        if switches.get('enable_video_gen') and "生成视频" in sentence:
            prompt = re.sub(r'生成视频[:：]?', '', sentence).strip()
            return generate_video(prompt)

        # ========== 收集所有分析结果 ==========
        analysis_results = {}
        
        # 模块1：文本分类
        if switches.get('enable_category') and sentence:
            try:
                from text_categorization import predict_text_category
                cat_label, cat_score = predict_text_category(
                    text=sentence,
                    model_path=TEXT_CATEGORY_MODEL_PATH,
                    vocab_dir=VOCAB_DIR
                )
                analysis_results['category'] = {
                    'label': cat_label,
                    'score': cat_score
                }
            except Exception as e:
                print(f"文本分类失败：{str(e)}")
        
        # 模块2：情感分析
        if switches.get('enable_sentiment') and sentence:
            try:
                from sentiment_analysis import predict_sentiment
                sentiment_label, sentiment_score = predict_sentiment(
                    text=sentence,
                    dicts=_sentiment_dicts,
                    model=_sentiment_model
                )
                analysis_results['sentiment'] = {
                    'label': sentiment_label,
                    'score': sentiment_score
                }
            except Exception as e:
                print(f"情感分析失败：{str(e)}")
        
        # 模块3：关键词提取（修复：正确保存）
        if switches.get('enable_keywords') and sentence and len(sentence) >= 1:
            try:
                from keyword_extraction import extract_keywords_hybrid
                keywords = extract_keywords_hybrid(sentence, topK=5)
                if keywords:
                    analysis_results['keywords'] = keywords  # 直接保存列表
            except Exception as e:
                print(f"关键词提取失败：{str(e)}")
        
        # 模块4：文本摘要（修复：正确保存）
        if switches.get('enable_summary') and sentence and len(sentence) >= 50:
            try:
                from text_summarization import extract_summary_textrank
                summary = extract_summary_textrank(sentence, ratio=0.4)
                if summary and summary != "文本过短，无法生成摘要":
                    analysis_results['summary'] = summary  # 直接保存文本
            except Exception as e:
                print(f"文本摘要失败：{str(e)}")
        
        # 模块5：命名实体识别（修复：正确保存）
        if switches.get('enable_ner') and sentence:
            try:
                from named_entity_recognition import extract_all_entities
                entities = extract_all_entities(sentence)
                # 过滤空实体
                filtered_entities = {k: v for k, v in entities.items() if v}
                if filtered_entities:
                    analysis_results['entities'] = filtered_entities  # 保存字典
            except Exception as e:
                print(f"命名实体识别失败：{str(e)}")
        
        # 模块6：机器翻译
        zh2en_pattern = re.compile(r'中译英[:：]?\s*(.+?)($|；|。|，|！|？)|翻译[:：]?\s*(.+?)($|；|。|，|！|？)', re.IGNORECASE)
        translate_match = zh2en_pattern.search(sentence) if sentence else None
        
        if switches.get('enable_translation') and translate_match and _translation_model_loaded:
            try:
                from machine_translation import machine_translate
                translate_text = translate_match.group(1) or translate_match.group(3)
                translate_text = translate_text.strip() if translate_text else sentence
                
                if translate_text:
                    end_punct = '。！？；，'
                    if not translate_text or translate_text[-1] not in end_punct:
                        translate_text += '。'
                    
                    translate_result = machine_translate(translate_text,
                                                         src_lang="zh",
                                                         tgt_lang="en")
                    
                    # 构建翻译回复
                    res_msg = f"<b>【中译英结果】</b><br>{translate_result}"
                    
                    # 追加分析结果
                    if analysis_results:
                        res_msg += "<br><br>" + format_analysis(analysis_results)
                    
                    return normalize_text(res_msg)
            except Exception as e:
                print(f"翻译失败：{str(e)}")
                return normalize_text(f"翻译服务暂时不可用<br>错误：{str(e)}")
        
        # ========== 正常对话逻辑 ==========
        user_content = []
        if sentence:
            user_content.append({"type": "text", "text": sentence})
        if image_paths and len(image_paths) > 0:
            for img_path in image_paths:
                base64_image = image_to_base64(img_path)
                if base64_image:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_image}
                    })
        
        if not user_content:
            return normalize_text("请输入消息或上传图片～")
        
        # 深度思考功能
        messages = [{"role": "user", "content": user_content}]
        if switches.get('enable_deep_think') and sentence:
            messages.insert(0, {
                "role": "system",
                "content": "请基于用户的问题进行深度、全面的分析，包括但不限于：1. 核心问题拆解；2. 多角度分析；3. 潜在逻辑；4. 具体解决方案/建议。回答需条理清晰、逻辑严谨。"
            })

        # 调用豆包API
        comp = client.chat.completions.create(
            model=ARK_MODEL,
            messages=messages
        )
        
        # 处理回复格式
        raw_res = comp.choices[0].message.content
        markdown_res = markdown_to_normal(raw_res)
        final_res = normalize_text(markdown_res)
        
        # 追加分析结果（统一显示）
        if analysis_results:
            final_res += "<br><br>" + format_analysis(analysis_results)
        
        # 深度思考标识
        if switches.get('enable_deep_think'):
            final_res = f'<div style="color:#2196F3; font-size:12px; margin-bottom:8px;">💡 已启用深度思考模式</div>' + final_res
        
        return final_res
        
    except Exception as e:
        return f"❌ 对话错误：{str(e)}"

# ========== Flask 路由 ==========
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

@app.route('/refresh_video_url', methods=['POST'])
def refresh_video_url():
    """刷新过期的视频URL"""
    try:
        task_id = request.form.get('task_id', '').strip()
        if not task_id:
            return jsonify({'text': "❌ 请输入有效的任务ID"})
        
        fresh_url = get_fresh_video_url(task_id)
        if fresh_url:
            res_html = (
                f'<b>✅ URL刷新成功</b><br>'
                f'<div style="color:#ff6600; font-size:12px; margin:8px 0;">⚠️ 新链接有效期24小时</div>'
                f'<a href="{fresh_url}" target="_blank" style="display:inline-block; padding:8px 15px; background:#4CAF50; color:white; border-radius:4px; text-decoration:none;">点击下载视频</a>'
            )
            return jsonify({'text': res_html})
        else:
            return jsonify({'text': "❌ 刷新失败：任务不存在/状态异常"})
    except Exception as e:
        return jsonify({'text': f"❌ 刷新URL失败：{str(e)}"})

@app.route('/message', methods=['POST'])
def reply():
    """修复版：支持所有功能开关"""
    msg = request.form.get('msg', '').strip()
    # 整合所有开关
    switches = {
        k: request.form.get(k) == 'true' 
        for k in ['enable_image_gen', 'enable_video_gen', 'enable_sentiment', 
                 'enable_deep_think', 'enable_category', 'enable_translation',
                 'enable_keywords', 'enable_summary', 'enable_ner']
    }
    
    # 处理上传的图片
    image_paths = []
    if 'images' in request.files:
        files = request.files.getlist('images')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = str(int(time.time() * 1000))
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_paths.append(filepath)
                print(f"✓ 图片已保存：{filepath}")
    
    if not msg and not image_paths:
        return jsonify({'text': normalize_text('请输入内容或上传图片～')})

    res_text = chat(msg, image_paths, **switches)
    
    # 清理临时文件
    for p in image_paths:
        try:
            if os.path.exists(p): 
                os.remove(p)
        except Exception as e:
            print(f"清理临时文件失败：{e}")
        
    return jsonify({'text': res_text})

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    init_models()
    print("\n✅ 服务启动成功：http://127.0.0.1:8808")
    print("⚠️  生产环境请关闭debug模式")
    app.run(host='127.0.0.1', port=8808, debug=False)