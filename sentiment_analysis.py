# 10.3.2 情感分析
import pandas as pd
import numpy as np
import jieba
import time
import os
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Input
from sklearn.model_selection import train_test_split
from sklearn import metrics


# ========== 数据预处理 ==========
def prepare_data():
    '''准备情感分析数据'''
    # 读取语料
    neg = pd.read_excel('../data/neg.xls', header=None, index_col=None)
    pos = pd.read_excel('../data/pos.xls', header=None, index_col=None)

    # 贴标签
    pos['mark'] = 1
    neg['mark'] = 0

    # 合并数据
    pn_all = pd.concat([pos, neg], ignore_index=True)
    pn_all[0] = pn_all[0].astype(str)

    # 分词
    cut_word = lambda x: list(jieba.cut(x))
    pn_all['words'] = pn_all[0].apply(cut_word)

    # 扩展语料（可选）
    try:
        comment = pd.read_excel('../data/sum.xls')
        comment = comment[comment['rateContent'].notnull()]
        comment['words'] = comment['rateContent'].apply(cut_word)
        pn_comment = pd.concat([pn_all['words'], comment['words']], ignore_index=True)
    except:
        pn_comment = pn_all['words']

    # 构建词典
    w = []
    for i in pn_comment:
        w.extend(i)
    dicts = pd.DataFrame(pd.Series(w).value_counts())
    dicts['id'] = list(range(1, len(dicts) + 1))

    # 词语向量化
    get_sent = lambda x: list(dicts['id'][x])
    pn_all['sent'] = pn_all['words'].apply(get_sent)

    # 填充/截断
    maxlen = 50
    pn_all['sent'] = list(sequence.pad_sequences(pn_all['sent'], maxlen=maxlen))

    # 划分数据集
    x_all = np.array(list(pn_all['sent']))
    y_all = np.array(list(pn_all['mark']))
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25)

    return x_train, x_test, y_train, y_test, dicts


# ========== 模型构建 ==========
def build_sentiment_model(vocab_size):
    '''构建LSTM情感分析模型'''
    model = Sequential()
    model.add(Input(shape=(50,)))
    model.add(Embedding(vocab_size + 1, 256))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


# ========== 训练函数 ==========
def train_sentiment_model():
    '''训练情感分析模型'''
    save_dir = '../tmp/'
    model_path = os.path.join(save_dir, 'sentiment_model.h5')
    dicts_path = os.path.join(save_dir, 'sentiment_dicts.csv')

    # 准备数据
    print("准备数据...")
    x_train, x_test, y_train, y_test, dicts = prepare_data()

    print(f'训练集形状：{x_train.shape}')
    print(f'测试集形状：{x_test.shape}')

    # 构建模型
    print("构建模型...")
    model = build_sentiment_model(len(dicts))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    # 训练
    print("开始训练...")
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=16, epochs=10, verbose=1)
    print(f'训练耗时：{int(time.time() - start_time)}秒')

    # 评估
    print("评估模型...")
    y_pred = model.predict(x_test).round().astype(int)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f'测试集准确率：{acc:.4f}')
    print('\n分类报告：')
    print(metrics.classification_report(y_test, y_pred,
                                        target_names=['负面', '正面']))
    print('\n混淆矩阵：')
    print(metrics.confusion_matrix(y_test, y_pred))

    # 保存模型和词典
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(model_path)
    dicts.to_csv(dicts_path, index=True)
    print(f"\n模型已保存至：{model_path}")
    print(f"词典已保存至：{dicts_path}")

    return model, dicts


# ========== 推理函数（供外部调用）==========
def load_sentiment_deps(model_path='../tmp/sentiment_model.h5',
                        dicts_path='../tmp/sentiment_dicts.csv'):
    '''加载情感分析依赖'''
    try:
        dicts = pd.read_csv(dicts_path, index_col=0)
        dicts.columns = ['count', 'id']
        model = load_model(model_path, compile=False)
        return dicts, model
    except Exception as e:
        print(f"加载情感分析依赖失败：{str(e)}")
        return None, None


def predict_sentiment(text, dicts=None, model=None, maxlen=50):
    """
    情感分析预测接口
    :param text: 待分析文本
    :param dicts: 词典（可选，避免重复加载）
    :param model: 模型（可选）
    :param maxlen: 最大长度
    :return: (情感标签, 置信度)
    """
    try:
        # 若未传入依赖，则加载
        if dicts is None or model is None:
            dicts, model = load_sentiment_deps()
            if dicts is None or model is None:
                return "neutral", 0.5

        # 文本预处理
        text = str(text).strip()
        words = list(jieba.cut(text))

        # 词转ID
        sent = []
        for word in words:
            if word in dicts.index:
                sent.append(dicts['id'][word])

        # 填充/截断
        sent_pad = sequence.pad_sequences([sent], maxlen=maxlen)

        # 预测
        pred_prob = model.predict(sent_pad, verbose=0)[0][0]

        # 情感判定
        if pred_prob >= 0.7:
            label = "positive"
        elif pred_prob <= 0.3:
            label = "negative"
        else:
            label = "neutral"

        score = round(float(pred_prob), 4)

        return label, score

    except Exception as e:
        print(f"情感分析预测失败：{str(e)}")
        return "neutral", 0.5


# ========== 主函数 ==========
if __name__ == '__main__':
    # 检查模型是否存在
    model_path = '../tmp/sentiment_model.h5'
    if not os.path.exists(model_path):
        print("模型不存在，开始训练...")
        train_sentiment_model()
    else:
        print(f"模型已存在：{model_path}")
        # 测试推理
        test_texts = [
            "这款手机太好用了，续航超久！",
            "质量太差了，用了一天就坏了。",
            "还行吧，功能一般。"
        ]
        dicts, model = load_sentiment_deps()
        for text in test_texts:
            label, score = predict_sentiment(text, dicts, model)
            print(f"\n测试文本：{text}")
            print(f"情感分析：{label}，置信度：{score}")