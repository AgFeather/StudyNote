import tensorflow as tf
import numpy as np

"""使用LSTM构建自动小说生成程序"""


def load_data(data_path='dataset/anna_text.txt'):
    """加载数据集"""
    with open(data_path, 'r') as file:
        text = file.read()
    vocabulary = list(set(text))
    print("There are {} characters in dataset".format(len(vocabulary)))
    int2char = {i:c for i, c in enumerate(vocabulary)}
    char2int = {c:i for i, c in int2char.items()}
    int_text = [char2int[c] for c in text]
    return int_text, int2char, char2int


def get_flags():
    flags = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('batch_size', 64, 'the number of training batch')
    tf.app.flags.DEFINE_integer('time_steps', 50, 'time step')
    tf.app.flags.DEFINE_integer('embed_dim', 512, 'dimesion of embedding')
    tf.app.flags.DEFINE_integer('num_units', 512, 'number of units in hidden layer')
    tf.app.flags.DEFINE_integer('num_epochs', 8, 'number of epoch')
    tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    tf.app.flags.DEFINE_string('model_save_path', 'trained_model/lstm.ckpt', 'the path to save trained model')
    return flags

class DeepWriterModel(object):
    def __init__(self, flags, num_chars):
        self.flags = flags
        self.num_chars = num_chars
        self.input_x, self.target_y, self.init_state,\
        self.accuracy, self.loss, self.optimizer = self.build_model()

    def build_model(self):
        input_x = tf.placeholder(tf.int64, [None, None], name='input_x')
        target_y = tf.placeholder(tf.int64, [None, None], name='target_y')

        embed_matrix = tf.get_variable('embed_x', [self.num_chars, self.flags.embed_dim],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        lstm_input = tf.nn.embedding_lookup(embed_matrix, input_x)

        cells = []
        for i in range(2):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.flags.num_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell)
            cells.append(cell)

        lstm = tf.nn.rnn_cell.MultiRNNCell(cells)
        init_state = lstm.zero_state(self.flags.batch_size, tf.float32)
        lstm_output, final_sate = tf.nn.dynamic_rnn(lstm, lstm_input, initial_state=init_state)

        fc_input = tf.reshape(lstm_output, [-1, self.flags.num_units])
        fc_weight = tf.get_variable('fc_weight', [self.flags.num_units, self.flags.num_units],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc_bias = tf.get_variable('fc_bias', [self.flags.num_units], initializer=tf.constant_initializer(0.0))
        fc_layer = tf.nn.relu(tf.matmul(fc_input, fc_weight) + fc_bias)

        output_weight = tf.get_variable('output_weight', [self.flags.num_units, self.num_chars],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        output_bias = tf.get_variable('output_bias', [self.num_chars],
                                      initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(fc_layer, output_weight) + output_bias

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(target_y, [-1])))
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(logits, axis=1), tf.reshape(target_y, [-1])
                ),
            tf.float32)
        )
        optimizer = tf.train.AdamOptimizer(self.flags.learning_rate)
        gradient_pair = optimizer.compute_gradients(loss)
        clip_gradient_pair = []
        for grad, var in gradient_pair:
            grad = tf.clip_by_value(grad, -5., 5.)
            clip_gradient_pair.append((grad, var))
        optimizer = optimizer.apply_gradients(clip_gradient_pair)

        return input_x, target_y, init_state, accuracy, loss, optimizer

    def train(self, data):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            global_step = 0
            for i in range(self.flags.num_epochs):
                data_generator = self.get_generator(data)
                for batch_x, batch_y in data_generator:
                    global_step += 1
                    feed_dict = {self.input_x:batch_x, self.target_y:batch_y}
                    show_loss, show_accuracy, _ = sess.run(
                        [self.loss, self.accuracy ,self.optimizer], feed_dict)
                    if global_step % 10 == 0:
                        print('Epoch:{}, global step:{}, loss:{:.4f}, accuracy:{:.2f}%'.
                              format(i+1, global_step, show_loss, show_accuracy*100))
                saver.save(sess, self.flags.model_save_path)


    def get_generator(self, data):
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










if __name__ == '__main__':
    int_text, int2char, char2int = load_data()
    # print(int_text[:20])
    flags = get_flags()
    deep_writter = DeepWriterModel(flags, len(int2char))
    deep_writter.train(int_text)