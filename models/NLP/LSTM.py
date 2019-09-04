import tensorflow as tf
import numpy as np
import pickle

"""使用LSTM构建自动小说生成程序"""


def load_data(data_path='dataset/anna_text.txt', save_dict=False):
    """加载数据集"""
    with open(data_path, 'r') as file:
        text = file.read()
    vocabulary = list(set(text))
    map_dict_path = 'map_dict.pkl'
    print("There are {} characters in dataset".format(len(vocabulary)))
    if not save_dict:
        print("load map dict from {}".format(map_dict_path))
        int2char, char2int = pickle.load(open(map_dict_path, 'rb'))
    else:
        print("save map dict to {}".format(map_dict_path))
        int2char = {i:c for i, c in enumerate(vocabulary)}
        char2int = {c:i for i, c in int2char.items()}
    int_text = [char2int[c] for c in text]
    if save_dict:
        pickle.dump([int2char, char2int], open(map_dict_path, 'wb'))
    return int_text, int2char, char2int


def get_flags():
    flags = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('batch_size', 100, 'the number of training batch')
    tf.app.flags.DEFINE_integer('time_steps', 100, 'time step')
    tf.app.flags.DEFINE_integer('embed_dim', 1024, 'dimesion of embedding')
    tf.app.flags.DEFINE_integer('num_units', 1024, 'number of units in hidden layer')
    tf.app.flags.DEFINE_integer('num_epochs', 100, 'number of epoch')
    tf.app.flags.DEFINE_integer('show_every_n', 500, 'show every n global training steps')
    tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    tf.app.flags.DEFINE_string('model_save_path', 'trained_model/lstm/', 'the path to save trained model')
    return flags

class DeepWriterModel(object):
    def __init__(self, flags, num_chars, is_training=True):
        self.flags = flags
        self.num_chars = num_chars
        if not is_training:
            self.flags.batch_size = 1
        self.graph, self.input_x, self.target_y, self.keep_prob, self.init_state,\
        self.final_state, self.prediction, self.accuracy, self.loss, \
        self.optimizer = self.build_model()

    def build_model(self):
        graph = tf.Graph()
        with graph.as_default():
            input_x = tf.placeholder(tf.int64, [None, None], name='input_x')
            target_y = tf.placeholder(tf.int64, [None, None], name='target_y')
            keep_prob = tf.placeholder(tf.float32)

            embed_matrix = tf.get_variable('embed_x', [self.num_chars, self.flags.embed_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            lstm_input = tf.nn.embedding_lookup(embed_matrix, input_x)

            cells = []
            for i in range(2):
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.flags.num_units)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
                cells.append(cell)

            lstm = tf.nn.rnn_cell.MultiRNNCell(cells)
            init_state = lstm.zero_state(self.flags.batch_size, tf.float32)
            lstm_output, final_state = tf.nn.dynamic_rnn(lstm, lstm_input, initial_state=init_state)

            fc_input = tf.reshape(lstm_output, [-1, self.flags.num_units])
            output_weight = tf.get_variable('output_weight', [self.flags.num_units, self.num_chars],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            output_bias = tf.get_variable('output_bias', [self.num_chars],
                                          initializer=tf.constant_initializer(0.0))
            logits = tf.matmul(fc_input, output_weight) + output_bias

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(target_y, [-1])))
            accuracy = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.argmax(logits, axis=1), tf.reshape(target_y, [-1])
                    ),
                tf.float32)
            )
            trainable_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars), 5.)
            optimizer = tf.train.AdamOptimizer(self.flags.learning_rate)
            optimizer = optimizer.apply_gradients(zip(grads, trainable_vars))
            # gradient_pair = optimizer.compute_gradients(loss)
            # clip_gradient_pair = []
            # for grad, var in gradient_pair:
            #     grad = tf.clip_by_value(grad, -5., 5.)
            #     clip_gradient_pair.append((grad, var))
            # optimizer = optimizer.apply_gradients(clip_gradient_pair)

        return graph, input_x, target_y, keep_prob, \
               init_state, final_state, \
               logits, accuracy, loss, optimizer

    def train(self, data):
        with self.graph.as_default():
            saver = tf.train.Saver(max_to_keep=self.flags.num_epochs)
            with tf.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())
                global_step = 0
                for epoch in range(self.flags.num_epochs):
                    data_generator = self.batch_generator(data)
                    new_state = sess.run([self.init_state])
                    for batch_x, batch_y in data_generator:
                        global_step += 1
                        feed_dict = {self.input_x:batch_x,
                                     self.target_y:batch_y,
                                     self.keep_prob:0.5,
                                     self.init_state:new_state}
                        show_loss, show_accuracy, new_state, _ = sess.run(
                            [self.loss, self.accuracy, self.final_state, self.optimizer], feed_dict)
                        if global_step % self.flags.show_every_n == 0:
                            print('Epoch:{}, global step:{}, loss:{:.4f}, accuracy:{:.2f}%'.
                                  format(epoch+1, global_step, show_loss, show_accuracy*100))

                    print("epoch:{} model saved".format(epoch))
                    saver.save(sess, self.flags.model_save_path + "epoch{}.ckpt".format(epoch))


    def batch_generator(self, data):
        total_batch = len(data) // (self.flags.batch_size * self.flags.time_steps)
        data = data[:total_batch*(self.flags.batch_size * self.flags.time_steps)]
        train_x = np.array(data)
        train_y = np.zeros_like(train_x)
        train_y[:-1], train_y[-1] = data[1:], data[1]
        train_x = train_x.reshape([self.flags.batch_size, -1])
        train_y = train_y.reshape([self.flags.batch_size, -1])

        for i in range(0, train_x.shape[1], self.flags.time_steps):
            batch_x = train_x[:, i:i+self.flags.time_steps]
            batch_y = train_y[:, i:i+self.flags.time_steps]
            yield batch_x, batch_y

    def generate(self, int2char, char2int, start_words="The ", total_words=2000):
        self.flags.batch_size = 1
        with self.graph.as_default():
            generated_text = start_words[:]
            int_start_words = [char2int[c] for c in start_words]
            with tf.Session(graph=self.graph) as sess:
                saver = tf.train.Saver()
                checkpoint = tf.train.latest_checkpoint(self.flags.model_save_path)
                saver.restore(sess=sess, save_path=checkpoint)
                new_state = sess.run(self.init_state)
                for i in int_start_words:
                    x = np.zeros((1, 1))
                    x[0, 0] = i
                    feed_dict = {
                        self.input_x:x,
                        self.keep_prob:1.0,
                        self.init_state:new_state,
                    }
                    new_state, prediction_list = sess.run([self.final_state, self.prediction], feed_dict)
                pred_int = self.pick_topk(prediction_list)
                generated_text += int2char[pred_int]

                while(len(generated_text) <= total_words):
                    x = np.zeros((1,1))
                    x[0, 0] = pred_int
                    feed_dict = {
                        self.input_x:x,
                        self.keep_prob:1.0,
                        self.init_state:new_state,
                    }
                    new_state, output_text = sess.run([self.final_state, self.prediction], feed_dict)
                    pred_int = self.pick_topk(output_text)
                    generated_text += int2char[pred_int]

        print(generated_text)
        return generated_text

    def pick_topk(self, probabilities, top_n=3):
        """
        从预测结果中选取前top_n个最可能的字符
        """
        probabilities = np.squeeze(probabilities)
        probabilities[np.argsort(probabilities)[:-top_n]] = 0 # 将除了top_n个预测值的位置都置为0
        probabilities = probabilities / np.sum(probabilities) # 归一化概率
        c = np.random.choice(self.num_chars, 1, p=probabilities)[0] # 随机选取一个字符
        return c




if __name__ == '__main__':
    int_text, int2char, char2int = load_data(save_dict=False)
    flags = get_flags()
    deep_writter = DeepWriterModel(flags, len(int2char), is_training=False)
    # deep_writter.train(int_text)
    deep_writter.generate(int2char, char2int)
