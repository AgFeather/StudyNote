import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter


"""使用TensorFlow构建Skip-gram模型搭建Word2Vec 词向量"""

def preprocess(text, freq=5):
    '''
    对文本进行预处理，text: 文本数据，freq: 词频阈值
    '''
    # 对文本中的符号进行替换
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # 删除低频词，减少噪音影响
    word_counts = Counter(words)
    string_words = [word for word in words if word_counts[word] > freq]

    # 构建word到int的映射表
    vocab_set = set(string_words)
    vocab2int = {c: i for i, c in enumerate(vocab_set)}
    int2vocab = {i: c for i, c in enumerate(vocab_set)}
    int_words = [vocab2int[i] for i in string_words]

    print("total words: {}".format(len(string_words)))
    print("unique words: {}".format(len(set(string_words))))

    return string_words, int_words, vocab2int, int2vocab


def load_text8(data_path='dataset/text8'):
    with open(data_path, 'r') as file:
        text = file.read()
    return text


def down_sample(int_words, t=1e-5, threshold=0.8):
    """对停用词进行采样，例如“the”， “of”以及“for”这类单词进行剔除。
    剔除这些单词以后能够加快我们的训练过程，同时减少训练过程中的噪音。"""
    # 统计单词出现频次
    int_word_counts = Counter(int_words)
    total_count = len(int_words)
    # 计算单词频率
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
    # 计算被删除的概率
    prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
    # 对单词进行采样
    train_words = [w for w in int_words if prob_drop[w] < threshold]
    return train_words


def get_batches(words, batch_size = 64, window_size=5):
    def get_targets(batch_words, idx):
        """返回input word的上下文单词列表"""
        start_index = idx - window_size if (idx - window_size) > 0 else 0
        end_index = idx +window_size if (idx + window_size) < len(batch_words) else len(batch_words)
        targets = list(set(batch_words[start_index:idx] + batch_words[idx+1:idx+end_index+1]))
        return targets

    # 删除掉尾部剩余的单词
    n_batches = len(words) // batch_size
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i)
            # 一个input word对应多个output word，将长度统一
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        x = np.array(x)  # .reshape((-1, 1))
        y = np.array(y).reshape((-1, 1))
        yield x, y


class SkipGram(object):
    def __init__(self,
                 num_vocab,
                 embed_dim=200,
                 n_sampled=100,
                 learning_rate=0.01):

        self.embed_dim = embed_dim
        self.num_vocab = num_vocab
        self.n_sampled = n_sampled
        self.learning_rate = learning_rate

        self.build_model()

    def build_input(self):
        inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')
        return inputs, labels

    def build_embedding(self, inputs):
        embedding_matrix = tf.get_variable(
            'embedding_matrix', [self.num_vocab, self.embed_dim],
            dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        embedding_rep = tf.nn.embedding_lookup(embedding_matrix, inputs)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_matrix), 1, keep_dims=True))
        normalized_embedding = embedding_matrix / norm
        return embedding_rep, normalized_embedding

    def bulid_loss(self, weight, bias, labels, embed_input):
        loss = tf.nn.sampled_softmax_loss(
            weight, bias, labels, embed_input, self.n_sampled, self.num_vocab)
        loss = tf.reduce_mean(loss)
        return loss

    def bulid_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def build_model(self):
        tf.reset_default_graph()
        self.inputs, self.labels = self.build_input()
        self.embed_inputs, self.normalized_embedding = self.build_embedding(self.inputs)
        self.similarity = tf.matmul(self.embed_inputs, tf.transpose(self.normalized_embedding))
        softmax_weight = tf.get_variable(
            'softmax_weight', [self.num_vocab, self.embed_dim],
            dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        softmax_bias = tf.get_variable(
            'softmax_bias', [self.num_vocab],
            dtype=tf.float32, initializer=tf.truncated_normal_initializer)
        self.loss = self.bulid_loss(softmax_weight, softmax_bias, self.labels, self.embed_inputs)
        self.optimizer = self.bulid_optimizer(self.loss)

    def train(self, train_data):
        generator = get_batches(train_data)
        num_epoches = 1
        show_every_n = 100
        save_every_n = 500
        global_step = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epoches):
                for batch_x, batch_y in generator:
                    global_step += 1
                    feed = {self.inputs: batch_x,
                            self.labels: batch_y}
                    show_loss, _ = sess.run([self.loss, self.optimizer], feed)
                    if global_step % show_every_n == 0:
                        print('epoch:{}/{}'.format(epoch, num_epoches),
                              'step:{}'.format(global_step),
                              'train loss: {:.2f}'.format(show_loss))
                    if global_step % save_every_n == 0:
                        saver.save(sess, 'checkpoints/e{}_s{}.ckpt'.format(epoch, global_step))
            saver.save(sess, 'checkpoints/latest.ckpt')

    def get_similarity(self, words, int2vocab, vocab2int):
        # debug and fix
        saver = tf.train.Saver()
        checkpoints = tf.train.latest_checkpoint('checkpoints/')
        simi_list = []
        with tf.Session() as sess:
            saver.load(sess, checkpoints)
            for word in words:
                word = vocab2int[word]
                feed = {self.inputs: word}
                show_similarity = sess.run(self.similarity, feed)
                nearest = show_similarity.argmax()
                nearest_word = int2vocab[nearest]
                simi_list.append((nearest_word, nearest))
        return simi_list


if __name__ == '__main__':
    text = load_text8()
    string_words, int_words, vocab2int, int2vocab = preprocess(text)
    train_words = down_sample(int_words)
    model = SkipGram(len(vocab2int))
    model.train(train_words)