# 10.4 基于Seq2Seq的机器翻译
import re
import io
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


# ========== 数据预处理 ==========
def preprocess_sentence(w):
    '''句子预处理'''
    w = re.sub(r'([?.!,])', r' \1 ', w)
    w = re.sub(r"[' ']+", ' ', w)
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples):
    '''创建数据集'''
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]
                  for l in lines[:num_examples]]
    return zip(*word_pairs)


def max_length(tensor):
    '''计算最大长度'''
    return max(len(t) for t in tensor)


def tokenize(lang):
    '''分词并转为序列'''
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    '''加载数据集'''
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# ========== 模型定义 ==========
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_size=None):
        # ✅ 支持动态batch_size
        if batch_size is None:
            batch_size = self.batch_sz
        return tf.zeros((batch_size, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    '''注意力机制'''

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    '''解码器'''

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


# ========== 训练函数 ==========
def train_translation_model():
    '''训练机器翻译模型'''
    # 配置参数
    path_to_file = '../data/en-ch.txt'
    num_examples = 2000
    BUFFER_SIZE = 2000
    BATCH_SIZE = 64
    embedding_dim = 256
    units = 1024
    EPOCHS = 50
    checkpoint_dir = '../tmp/training_checkpoints'

    # 加载数据
    print("加载数据...")
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
        path_to_file, num_examples)

    max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

    # 划分训练集和验证集
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
        train_test_split(input_tensor, target_tensor, test_size=0.2)

    # 创建数据集
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # 构建模型
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    # 优化器和损失
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    # 训练步骤
    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = loss / int(targ.shape[1])
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    # 检查点
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    # 开始训练
    print("开始训练...")
    loss_history = []

    for epoch in tqdm(range(EPOCHS)):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state(batch_size=BATCH_SIZE)  # ✅ 明确指定batch_size
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')
                loss_history.append(round(float(batch_loss.numpy()), 3))

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, 'ckpt'))

        print(f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}')
        print(f'Time taken: {time.time() - start:.2f} sec\n')

    # 可视化损失
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(loss_history)
    plt.title('损失趋势图', fontsize=16)
    plt.xlabel('训练批次')
    plt.ylabel('损失值')
    plt.tight_layout()
    plt.savefig(os.path.join('../tmp/', 'translation_loss.png'))
    plt.show()

    print("训练完成！")
    return encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ


# ========== 推理函数（供外部调用）==========
# 全局变量（缓存加载的模型）
_encoder = None
_decoder = None
_inp_lang = None
_targ_lang = None
_max_length_inp = None
_max_length_targ = None
_units = 1024
_embedding_dim = 256


def load_translation_model():
    '''加载翻译模型（系统启动时调用一次）'''
    global _encoder, _decoder, _inp_lang, _targ_lang, _max_length_inp, _max_length_targ

    try:
        print("加载翻译模型...")
        # 加载数据集（仅为获取tokenizer）
        path_to_file = '../data/en-ch.txt'
        num_examples = 2000

        input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer = \
            load_dataset(path_to_file, num_examples)

        _max_length_targ = max_length(target_tensor)
        _max_length_inp = max_length(input_tensor)
        _inp_lang = inp_lang_tokenizer
        _targ_lang = targ_lang_tokenizer

        # 构建模型结构
        vocab_inp_size = len(_inp_lang.word_index) + 1
        vocab_tar_size = len(_targ_lang.word_index) + 1

        # 注意：推理时batch_size=1
        _encoder = Encoder(vocab_inp_size, _embedding_dim, _units, 1)
        _decoder = Decoder(vocab_tar_size, _embedding_dim, _units, 1)

        # 加载检查点
        checkpoint_dir = '../tmp/training_checkpoints'
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(),
                                         encoder=_encoder,
                                         decoder=_decoder)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

        print("翻译模型加载成功！")

    except Exception as e:
        print(f"翻译模型加载失败：{str(e)}")


def machine_translate(sentence, src_lang="zh", tgt_lang="en"):
    """机器翻译接口"""
    if src_lang != "zh" or tgt_lang != "en":
        return f"暂仅支持中译英"

    try:
        if _encoder is None or _decoder is None:
            return "翻译模型未加载，请先调用load_translation_model()"

        # 预处理
        sentence = preprocess_sentence(sentence)
        inputs = [_inp_lang.word_index.get(i, 0) for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=_max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        # ✅ 使用动态batch_size初始化
        hidden = _encoder.initialize_hidden_state(batch_size=1)
        enc_out, enc_hidden = _encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([_targ_lang.word_index['<start>']], 0)

        # 逐词预测
        result = ''
        for t in range(_max_length_targ):
            predictions, dec_hidden, _ = _decoder(dec_input, dec_hidden, enc_out)
            predicted_id = tf.argmax(predictions[0]).numpy()

            predicted_word = _targ_lang.index_word.get(predicted_id, '<unk>')
            if predicted_word == '<end>':
                break

            result += predicted_word + ' '
            dec_input = tf.expand_dims([predicted_id], 0)

        return result.strip()

    except Exception as e:
        return f"翻译失败：{str(e)}"


# ========== 主函数 ==========
if __name__ == '__main__':
    # 检查检查点是否存在
    checkpoint_dir = '../tmp/training_checkpoints'
    if not os.path.exists(checkpoint_dir) or not tf.train.latest_checkpoint(checkpoint_dir):
        print("检查点不存在，开始训练...")
        train_translation_model()

    # 测试推理
    print("\n测试翻译功能...")
    load_translation_model()

    test_sentences = ['我生病了。', '为什么不？', '让我一个人呆会儿。']
    for sent in test_sentences:
        result = machine_translate(sent)
        print(f"输入：{sent}")
        print(f"翻译：{result}\n")