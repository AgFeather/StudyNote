'''
使用TensorFlow构建最常见的DNN，将学习到的TensorFlow技巧不断添加进来

1. 模型的保存和加载
2. tensorboard
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

class DeepNeuralNetwork():
    def __init__(self, flags):
        self.FLAGS = flags
        self.build_model()

    def build_model(self):
        self.input_x = tf.placeholder(tf.float32, [None, 784], name='input_x')
        self.target_y = tf.placeholder(tf.float32, [None, 10], name='target_y')
        self.keep_drop = tf.placeholder(tf.float32)

        weight1 = tf.Variable(tf.random_normal(
            [784, self.FLAGS.num_hidden_units], dtype=tf.float32), name='weight1')
        bias1 = tf.Variable(tf.constant(0.1, tf.float32, [self.FLAGS.num_hidden_units]), name='bias1')
        layer1 = tf.nn.relu(tf.matmul(self.input_x, weight1) + bias1)

        weight2 = tf.Variable(tf.random_normal(
            [self.FLAGS.num_hidden_units, self.FLAGS.num_hidden_units], dtype=tf.float32), name='weight2')
        bias2 = tf.Variable(tf.constant(0.1, tf.float32, [self.FLAGS.num_hidden_units]), name='bias2')
        layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + bias2)

        output_weight = tf.Variable(tf.random_normal(
            [self.FLAGS.num_hidden_units, 10], dtype=tf.float32), name='output_weight')
        output_bias = tf.Variable(tf.constant(0.1, tf.float32, [10]), name='output_bias')
        # output_layer = tf.nn.relu(tf.matmul(layer2, output_weight) + output_bias) # 最后一层不能加relu
        output_layer = tf.matmul(layer2, output_weight) + output_bias

        self.softmax_output = tf.nn.softmax(output_layer)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=self.target_y)
        self.loss = tf.reduce_mean(self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(
                tf.argmax(output_layer, axis=1), tf.argmax(self.target_y, axis=1)
            ), tf.float32), name='accuracy')

        tf.summary.histogram('weight1', weight1)
        tf.summary.histogram('bias1', bias1)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def train(self):
        saver = tf.train.Saver()
        batch_size = 64
        num_epochs = 1
        global_step = 1
        each_step = 2000
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(self.FLAGS.tensorboard_log_path, sess.graph)
            for epoch in range(num_epochs):
                for __ in range(each_step):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    feed_dict = {self.input_x: batch_x, self.target_y: batch_y}
                    accu, loss, _, summary_str = sess.run(
                        [self.accuracy, self.loss, self.optimizer, self.merged], feed_dict)
                    writer.add_summary(summary_str, global_step)
                    if global_step % 500 == 0:
                        print('Epoch: {}; Global Step: {}; accuracy: {:.2f}; loss: {:.2f}'.
                              format(epoch + 1, global_step, accu, loss))

                    global_step += 1
                saver.save(sess, self.FLAGS.model_save_path)

    def eval(self):
        new_saver = tf.train.Saver()
        #checkpath = tf.train.latest_checkpoint()
        #print(checkpath)
        with tf.Session() as new_sess:
            #获取参数到new_sess 中
            graph = tf.get_default_graph()
            new_saver.restore(new_sess, self.FLAGS.model_save_path)
            input_x = graph.get_tensor_by_name('input_x:0')
            output_y = graph.get_tensor_by_name('target_y:0')
            accuracy = graph.get_tensor_by_name('accuracy:0')
            feed = {input_x:mnist.test.images, output_y:mnist.test.labels}
            accu = new_sess.run(accuracy, feed_dict=feed)
            print('test accuarcy:{:.4f}%'.format(accu*100))




FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_save_path', 'trained_model/dnn_model.ckpt', 'model_save_path')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning_rate')
tf.app.flags.DEFINE_integer('num_hidden_units', 1024, 'number of hidden units')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'number of epoch to train')
tf.app.flags.DEFINE_string('tensorboard_log_path', 'tensorboard_log/', 'tensorboard_log_path')


if __name__ == '__main__':
    model = DeepNeuralNetwork(FLAGS)
    #model.train()
    model.eval()
