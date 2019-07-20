import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

"""使用TensorFlow实现一个基础的GAN神经网络, 数据集为MNIST"""

def get_flags():
    flags = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('batch_size', 64, 'the number of training batch')
    tf.app.flags.DEFINE_integer('num_units', 256, 'number of units in hidden layer')
    tf.app.flags.DEFINE_integer('num_epochs', 10, 'number of epoch')
    tf.app.flags.DEFINE_integer('show_every_n', 100, 'show every n global training steps')
    tf.app.flags.DEFINE_integer('noise_size', 100, 'size of noise image')
    tf.app.flags.DEFINE_integer('real_size', 784, 'the size of true image')
    tf.app.flags.DEFINE_integer('num_sample', 25, 'number of sample image per training epoch')

    tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    tf.app.flags.DEFINE_float('smooth', 0.1, 'smooth')

    tf.app.flags.DEFINE_string('model_save_path', 'trained_model/gan/', 'the path to save trained model')
    return flags


class MnistGAN(object):
    """一个MNIST的GAN神经网络，仅使用最简单的GAN模型结构"""
    def __init__(self, flags):
        self.flags = flags
        self.graph, self.real_image_input, self.noise_image_input, \
        self.sample_logits, self.sample_outputs, \
        self.g_loss, self.d_loss, \
        self.disc_optimizer, self.gen_optimizer = self.build_model()

    def build_model(self):
        graph = tf.Graph()
        with graph.as_default():
            noise_image_input = tf.placeholder(tf.float32, [None, self.flags.noise_size], name='noise_img')
            real_image_input = tf.placeholder(tf.float32, [None, self.flags.real_size], name='real_img')

            with tf.variable_scope("generator", reuse=False):
                """构建图片生成器generator，
                将noise image输入到一个小型神经网络中，
                输出即为generator生成的输出图片
                神经网络的输出大小为真实图片的大小"""
                hidden1 = tf.layers.dense(noise_image_input, self.flags.num_units, activation=tf.nn.relu)
                gen_logits = tf.layers.dense(hidden1, self.flags.real_size)
                gen_outputs = tf.tanh(gen_logits)

            with tf.variable_scope("generator", reuse=True):
                """生成一个图片，在模型训练结束后进行生成时使用"""
                hidden1 = tf.layers.dense(noise_image_input, self.flags.num_units, activation=tf.nn.relu)
                sample_logits = tf.layers.dense(hidden1, self.flags.real_size)
                sample_outputs = tf.tanh(sample_logits)

            with tf.variable_scope("discriminator", reuse=False):
                """构建一个判别器discriminator，输入real image，并让判别器给real image打分"""
                hidden1 = tf.layers.dense(real_image_input, self.flags.num_units, activation=tf.nn.relu)
                d_logits_real = tf.layers.dense(hidden1, 1)
                d_outputs_real = tf.sigmoid(d_logits_real)

            with tf.variable_scope("discriminator", reuse=True):
                """让判别器对生成器generator生成的图片进行判别打分"""
                hidden1 = tf.layers.dense(gen_outputs, self.flags.num_units, activation=tf.nn.relu)
                d_logits_fake = tf.layers.dense(hidden1, 1)
                d_outputs_fake = tf.sigmoid(d_logits_fake)


            # discriminator的loss，
            # 对real image，所有标签都为1，也就是说令logits逼近1
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_real, labels=tf.ones_like(d_logits_real)) * (1 - self.flags.smooth))
            # 对生成器生成的image，所有标签为0，也就是说令logits逼近0
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
            d_loss = tf.add(d_loss_real, d_loss_fake)

            # generator的loss，让生成的假图片逼近1
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)) * (1 - self.flags.smooth))

            # 分别对两个神经网络进行optimizer
            train_vars = tf.trainable_variables()
            g_vars = [var for var in train_vars if var.name.startswith("generator")]  # generator中的tensor
            d_vars = [var for var in train_vars if var.name.startswith("discriminator")]  # discriminator中的tensor
            disc_optimizer = tf.train.AdamOptimizer(self.flags.learning_rate).minimize(d_loss, var_list=d_vars)
            gen_optimizer = tf.train.AdamOptimizer(self.flags.learning_rate).minimize(g_loss, var_list=g_vars)

        return graph, real_image_input, noise_image_input, \
               sample_logits, sample_outputs, \
               g_loss, d_loss, \
               disc_optimizer, gen_optimizer

    def train(self, mnist):
        print('model training begin...')
        samples = []  # 存储测试样例
        with self.graph.as_default():
            train_vars = tf.trainable_variables()
            # generator中的tensor
            g_vars = [var for var in train_vars if var.name.startswith("generator")]
            saver = tf.train.Saver(var_list=g_vars)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for e in range(self.flags.num_epochs):
                    for batch_i in range(mnist.train.num_examples // self.flags.batch_size):
                        batch = mnist.train.next_batch(self.flags.batch_size)
                        real_images = batch[0].reshape((-1, 784))  # 舍弃label，只用image
                        real_images = real_images * 2 - 1  # 将图像像素scale到(-1,1)之间
                        # 随机初始化一个array，作为generator的输入噪声
                        noise_images = np.random.uniform(-1, 1, size=(self.flags.batch_size, self.flags.noise_size))
                        feed_dict = {
                            self.real_image_input:real_images,
                             self.noise_image_input:noise_images,
                        }

                        _, __ = sess.run([self.disc_optimizer, self.gen_optimizer], feed_dict)

                    # 每一轮结束计算loss
                    show_d_loss, show_g_loss = sess.run([self.d_loss, self.g_loss], feed_dict)
                    print("Epoch {}, d_loss:{:.2f}, g_loss:{:.2f}".format(e + 1, show_d_loss, show_g_loss))

                    # 运行完每一个epoch后用generator生成图片，并保存
                    sample_noise = np.random.uniform(-1, 1, size=(self.flags.num_sample, self.flags.noise_size))
                    one_epoch_samples = sess.run([self.sample_logits, self.sample_outputs],
                                                 feed_dict={self.noise_image_input: sample_noise})
                    samples.append(one_epoch_samples)

                    saver.save(sess, self.flags.model_save_path + 'epoch{}.ckpt'.format(e))

        # 保存生成的图片
        pickle.dump(samples, open('train_sample.pkl', 'wb'))


class ImageGenerator(object):
    """加载已经训练完成的模型，并可以用该模型的generator进行图片生成"""

    def __init__(self, flags):
        self.flags = flags
        self.model = MnistGAN(flags)

    def display_training_samples(self):
        """读取训练时各个epoch生成的sample并显示"""
        with open('train_sample.pkl', 'rb') as file:
            samples = pickle.load(file)
        for epoch in range(self.flags.num_epochs):
            self.view_samples(epoch, samples)
        plt.show()

    def view_samples(self, epoch, samples):
        """epoch代表第几次迭代的图像, samples为我们的采样结果"""
        fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples[epoch][1]):  # samples[epoch][1]代表生成的图像结果，[0]代表logits
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
            #plt.show()
        return fig, axes

    def generate(self, noise_size=100):
        """加载模型，并生成新的图片"""
        saver = tf.train.Saver()
        with tf.Session() as sess:
            checkpoints = tf.train.latest_checkpoint('checkpoints/')
            saver.restore(sess, checkpoints)
            sample_noise = np.random.uniform(-1, 1, size=(25, noise_size))
            gen_samples = self.session.run([self.model.sample_logits, self.model.sample_outputs],
                                           feed_dict={self.model.noise_image_input: sample_noise})
            _ = self.view_samples(0, [gen_samples])


if __name__ == '__main__':
    flags =get_flags()
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    # model = MnistGAN(flags)
    # model.train(mnist)
    image_generator = ImageGenerator(flags)
    image_generator.display_training_samples()
