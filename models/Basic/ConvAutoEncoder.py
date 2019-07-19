import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

"""使用卷积自编码机实现图片的降噪功能"""

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)


def ConvAutoEncoder():
    input_x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_x')
    label_y = tf.placeholder(tf.float32, [None, 28, 28, 1], name='label_y')

    # Encoder: three conv_layer
    conv1 = tf.layers.conv2d(input_x, 64, (3, 3), strides=1, padding='same', activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, (2, 2), strides=(2, 2), padding='same')
    conv2 = tf.layers.conv2d(conv1, 64, (3, 3), strides=1, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, (2, 2), strides=(2, 2), padding='same')
    conv3 = tf.layers.conv2d(conv2, 32, (3, 3), strides=1, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.max_pooling2d(conv3, (2, 2), strides=(2, 2), padding='same')

    # Decoder:
    conv4 = tf.image.resize_nearest_neighbor(conv3, (7, 7))
    conv4 = tf.layers.conv2d(conv4, 32, (3, 3), strides=1, padding='same', activation=tf.nn.relu)
    conv5 = tf.image.resize_nearest_neighbor(conv4, (14, 14))
    conv5 = tf.layers.conv2d(conv5, 64, (3, 3), strides=1, padding='same', activation=tf.nn.relu)
    conv6 = tf.image.resize_nearest_neighbor(conv5, (28, 28))
    conv6 = tf.layers.conv2d(conv6, 64, (3, 3), strides=1, padding='same', activation=tf.nn.relu)

    logits = tf.layers.conv2d(conv6, 1, kernel_size=(3, 3), strides=1, padding='same', activation=None)
    output_y = tf.nn.sigmoid(logits, name='output_y')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_y, logits=logits))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    return input_x, label_y, optimizer, output_y, loss


def train():
    input_x, label_y, optimizer, output_y, loss = ConvAutoEncoder()
    noise_factor = 0.5
    epochs = 1
    batch_size = 128
    global_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for i in range(mnist.train.num_examples // batch_size):
                global_step += 1
                images = mnist.train.next_batch(batch_size)[0].reshape((-1, 28, 28, 1))

                noisy_images = images + noise_factor * np.random.randn(*images.shape)
                noisy_images = np.clip(noisy_images, 0., 1)
                feed_dict = {
                    input_x:noisy_images,
                    label_y:images,
                }
                show_loss, _ = sess.run([loss, optimizer], feed_dict)
                if global_step % 100 == 0:
                    print('epoch:{}, global step:{}, loss:{:.4f}'.format(e+1, global_step, show_loss))

            # 测试模型，可视化：
            fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20, 4))
            in_imgs = mnist.test.images[10:20]
            noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)

            reconstructed = sess.run(output_y,
                                     feed_dict={input_x: noisy_imgs.reshape((10, 28, 28, 1))})

            for images, row in zip([in_imgs, noisy_imgs, reconstructed], axes):
                for img, ax in zip(images, row):
                    ax.imshow(img.reshape((28, 28)))
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

            fig.tight_layout(pad=0.1)
            plt.show()



if __name__ == '__main__':
    train()






