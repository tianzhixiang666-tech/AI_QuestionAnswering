# 10.3.1 文本分类
import tensorflow as tf
from collections import Counter
from tensorflow import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# ========== 工具函数 ==========
def open_file(filename, mode='r'):
    '''打开文件'''
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    '''读取文件数据'''
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    '''构建词汇表'''
    data_train, _ = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    '''读取词汇表'''
    with open_file(vocab_dir) as fp:
        words = [i.strip() for i in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    '''读取分类目录'''
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    '''将文件转换为id表示'''
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    return x_pad, y_pad


# ========== 模型构建 ==========
def TextRNN(vocab_size):
    '''构建LSTM文本分类模型'''
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, 128, input_length=600),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.BatchNormalization(epsilon=1e-6, axis=1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


# ========== 训练函数 ==========
def train_model():
    '''训练文本分类模型'''
    # 数据路径配置
    base_dir = '../data/'
    train_dir = os.path.join(base_dir, 'cnews.train.txt')
    test_dir = os.path.join(base_dir, 'cnews.test.txt')
    val_dir = os.path.join(base_dir, 'cnews.val.txt')
    vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
    save_dir = '../tmp/'
    model_path = os.path.join(save_dir, 'text_category_model.h5')

    # 构建词汇表
    vocab_size = 5000
    if not os.path.exists(vocab_dir):
        print("构建词汇表...")
        build_vocab(train_dir, vocab_dir, vocab_size)

    # 读取分类和词汇表
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    vocab_size = len(words)

    # 加载数据
    seq_length = 600
    print("加载训练数据...")
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)

    # 构建模型
    print("构建模型...")
    model = TextRNN(vocab_size)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['categorical_accuracy'])

    # 训练模型
    print("开始训练...")
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=20,
                        validation_data=(x_val, y_val),
                        verbose=1)

    # 保存模型
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(model_path)
    print(f"模型已保存至：{model_path}")

    # 可视化训练过程
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'SimHei'

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.title('准确率趋势图')
    plt.plot(range(1, 21), history.history['categorical_accuracy'],
             linestyle='-', color='g', label='训练集')
    plt.plot(range(1, 21), history.history['val_categorical_accuracy'],
             linestyle='-.', color='b', label='验证集')
    plt.legend(loc='best')
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')

    plt.subplot(122)
    plt.title('损失趋势图')
    plt.plot(range(1, 21), history.history['loss'],
             linestyle='-', color='g', label='训练集')
    plt.plot(range(1, 21), history.history['val_loss'],
             linestyle='-.', color='b', label='验证集')
    plt.legend(loc='best')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'text_category_training.png'))
    plt.show()

    # 测试模型
    print("评估模型...")
    y_pred = model.predict(x_test)
    from sklearn.metrics import confusion_matrix, classification_report

    print("\n分类报告：")
    print(classification_report(np.argmax(y_test, axis=1),
                                np.argmax(y_pred, axis=1),
                                target_names=categories))

    # 混淆矩阵可视化
    confm = confusion_matrix(np.argmax(y_test, axis=1),
                             np.argmax(y_pred, axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(confm.T, square=True, annot=True, fmt='d',
                cbar=False, linewidths=.8, cmap='YlGnBu')
    plt.xlabel('真实标签', size=14)
    plt.ylabel('预测标签', size=14)
    plt.xticks(np.arange(10) + 0.5, categories, size=12)
    plt.yticks(np.arange(10) + 0.3, categories, size=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'text_category_confusion_matrix.png'))
    plt.show()

    return model


# ========== 推理函数（供外部调用）==========
def predict_text_category(text,
                          model_path='../tmp/text_category_model.h5',
                          vocab_dir='../data/cnews.vocab.txt',
                          max_length=600):
    """
    文本分类预测接口
    :param text: 待分类文本
    :param model_path: 模型路径
    :param vocab_dir: 词汇表路径
    :param max_length: 文本最大长度
    :return: (分类标签, 置信度)
    """
    try:
        # 加载依赖
        from tensorflow.keras.models import load_model

        categories, _ = read_category()
        words, word_to_id = read_vocab(vocab_dir)

        # 加载模型
        model = load_model(model_path, compile=False)

        # 预处理文本
        content = list(text)
        data_id = [word_to_id[x] for x in content if x in word_to_id]
        x_pad = keras.preprocessing.sequence.pad_sequences([data_id], max_length)

        # 预测
        pred_probs = model.predict(x_pad, verbose=0)
        pred_idx = np.argmax(pred_probs[0])
        pred_label = categories[pred_idx]
        pred_score = round(float(pred_probs[0][pred_idx]), 4)

        # 释放资源
        del model
        tf.keras.backend.clear_session()

        return pred_label, pred_score

    except Exception as e:
        print(f"文本分类预测失败：{str(e)}")
        return "未知", 0.0


# ========== 主函数 ==========
if __name__ == '__main__':
    # 检查模型是否存在
    model_path = '../tmp/text_category_model.h5'
    if not os.path.exists(model_path):
        print("模型不存在，开始训练...")
        train_model()
    else:
        print(f"模型已存在：{model_path}")
        # 测试推理
        test_text = "华为发布新款Mate60手机，搭载麒麟芯片"
        label, score = predict_text_category(test_text)
        print(f"\n测试文本：{test_text}")
        print(f"分类结果：{label}，置信度：{score}")