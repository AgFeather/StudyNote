import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 定义数据相关常量
IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_LABELS = 1000

# 第一层卷积层
CONV1_SIZE = 11
CONV1_DEEP = 96
CONV1_STRIDE = 4
POOL1_SIZE = 3
POOL1_STRIDE = 2


# 第二层卷积层
CONV2_SIZE = 5
CONV2_DEEP = 256
CONV2_STRIDE = 1
POOL2_SIZE = 3
POOL2_STRIDE = 2

# 第三到五卷积层
CONV3_SIZE = 3
CONV3_DEEP = 384
CONV4_SIZE = 3
CONV4_DEEP = 384
CONV5_SIZE = 3
CONV5_DEEP = 256
POOL5_SIZE = 3
POOL5_STRIDE = 2

# 全连接层的节点个数
FC_SIZE = 4096
FC_OUTPUT = 1000


class AlexNetModel():
    def __init__(self, flags):
        self.flags = flags
        self.build_model()
        self.build_optimizer()

    def build_model(self):
        self.input_x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='input_x')
        self.target_y = tf.placeholder(tf.float32, [None, NUM_LABELS], name='target_y')
        self.keep_drop = tf.placeholder(tf.float32)

        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable('weight',
                                            [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable('bias',
                                           [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

            conv1 = tf.nn.conv2d(self.input_x, conv1_weights, strides=[1, CONV1_STRIDE, CONV1_STRIDE, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
            lrn1 = tf.nn.lrn(relu1)
            pool1 = tf.nn.max_pool(lrn1, ksize=[1, POOL1_SIZE, POOL1_SIZE, 1],
                                   strides=[1, POOL1_STRIDE, POOL1_STRIDE, 1], padding='SAME')

        with tf.variable_scope('layer2-conv2'):
            conv2_weights = tf.get_variable('weight',
                                            [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_baises = tf.get_variable('bias',
                                           [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, CONV2_STRIDE, CONV2_STRIDE, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_baises))
            lrn2 = tf.nn.lrn(relu2)
            pool2 = tf.nn.max_pool(lrn2, ksize=[1, POOL2_SIZE, POOL2_SIZE, 1],
                                   strides=[1, POOL2_STRIDE, POOL2_STRIDE, 1], padding='SAME')

        with tf.variable_scope('layer3-conv3'):
            conv3_weights = tf.get_variable('weight',
                                            [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv3_baises = tf.get_variable('bias',
                                           [CONV3_DEEP], initializer=tf.constant_initializer(0.0))

            conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_baises))

        with tf.variable_scope('layer4-conv4'):
            conv4_weights = tf.get_variable('weight',
                                            [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv4_baises = tf.get_variable('bias',
                                           [CONV4_DEEP], initializer=tf.constant_initializer(0.0))

            conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_baises))

        with tf.variable_scope('layer5-conv5'):
            conv5_weights = tf.get_variable('weight',
                                            [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv5_baises = tf.get_variable('bias',
                                           [CONV5_DEEP], initializer=tf.constant_initializer(0.0))

            conv5 = tf.nn.conv2d(relu4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_baises))
            pool5 = tf.nn.max_pool(relu5, ksize=[1, POOL5_SIZE, POOL5_SIZE, 1],
                                   strides=[1, POOL5_STRIDE, POOL5_STRIDE, 1])



        map_shape = pool5.get_shape().as_list()
        flatten_size = map_shape[1] * map_shape[2] * map_shape[3]
        flatting = tf.reshape(pool5, [-1, flatten_size])

        with tf.variable_scope('layer6-fc1'):
            fc1_weights = tf.get_variable('weight',
                                          [flatten_size, FC_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc1_biases = tf.get_variable('bias',
                                         [FC_SIZE],
                                         initializer=tf.constant_initializer(0.0))
            fc1 = tf.nn.relu(tf.matmul(flatting, fc1_weights) + fc1_biases)
            fc1 = tf.nn.dropout(fc1, self.keep_drop)

        with tf.variable_scope('layer7-fc2'):
            fc2_weights = tf.get_variable('weight', [FC_SIZE, FC_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc2_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.0))
            fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
            fc2 = tf.nn.dropout(fc2, self.keep_drop)

        with tf.variable_scope('layer7-fc3'):
            fc3_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            fc3_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
            self.logits = tf.matmul(fc2, fc3_weights) + fc3_biases

    def build_optimizer(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.target_y), name='loss')
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(self.logits, axis=1),
            tf.argmax(self.target_y, axis=1)), tf.float32), name='accuracy')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merge_op = tf.summary.merge_all()

    def train(self, mnist):
        saver = tf.train.Saver()
        global_step = 0
        each_step = 500
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(self.flags.tensorboard_log_path, sess.graph)
            for epoch in range(self.flags.num_epochs):
                for _ in range(each_step):
                    global_step += 1
                    batch_x, batch_y = mnist.train.next_batch(self.flags.batch_size)
                    feed_dict = {self.input_x: batch_x, self.target_y: batch_y, self.keep_drop:0.5}
                    accu, loss, _, summary_str = sess.run([self.accuracy, self.loss, self.optimizer, self.merge_op],
                                             feed_dict)
                    writer.add_summary(summary_str, global_step)
                    if global_step % 50 == 0:
                        print('Epoch: {}; Global Step: {}; accuracy: {:.2f}%; loss: {:.4f}'.
                              format(epoch + 1, global_step, accu*100, loss))
                saver.save(sess, self.flags.model_save_path)
                print('trained model has been saved in epoch:{}'.format(epoch+1))

    def eval(self, mnist):
        new_saver = tf.train.Saver()
        with tf.Session() as new_sess:
            # 获取参数到new_sess 中
            graph = tf.get_default_graph()
            new_saver.restore(new_sess, self.flags.model_save_path)
            input_x = graph.get_tensor_by_name('input_x:0')
            output_y = graph.get_tensor_by_name('target_y:0')
            accuracy = graph.get_tensor_by_name('accuracy:0')
            feed = {input_x: mnist.test.images, output_y: mnist.test.labels, self.keep_drop:1.0}
            accu = new_sess.run(accuracy, feed_dict=feed)
            print('test accuarcy:{:.2f}%'.format(accu * 100))




if __name__ == '__main__':

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    FLAGS = tf.app.flags.FLAGS
    lenet_model = AlexNetModel(FLAGS)

    tf.app.flags.DEFINE_string('model_save_path', 'trained_model/lenet/lenet_model.ckpt', 'model_save_path')
    tf.app.flags.DEFINE_float('learning_rate', 0.002, 'learning_rate')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'number of batch size')
    tf.app.flags.DEFINE_integer('num_epochs', 1, 'number of epoch to train')
    tf.app.flags.DEFINE_string('tensorboard_log_path', 'tensorboard_log/lenet/', 'tensorboard_log_path')

    lenet_model.train(mnist)
    lenet_model.eval(mnist)