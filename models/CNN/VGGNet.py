import tensorflow as tf


batch_size = 32
num_batches = 100

# 用来创建卷积层并把本层的参数存入参数列表
# input_tensor:输入的tensor name:该层的名称, out_channel:输出通道数，p是参数列表
def conv_layer(input_tensor,name,out_channel,parameter_list):
    # 输入的通道数
    in_channel = input_tensor.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "weight", shape=[3,3,in_channel,out_channel],
                                 dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(input_tensor, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variable(scope + "bias", [out_channel], initializer=tf.constant_initializer(0.0))
        relu = tf.nn.relu(tf.nn.bias_add(conv,biases))
        parameter_list += [kernel,biases]
    return relu

# 定义最大池化层
def maxpool_layer(input_tensor,name):
    return tf.nn.max_pool(input_tensor, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)

# 定义全连接层
def fc_layer(input_tensor, name, out_channel, parameter_list):
    in_channel = input_tensor.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope+"weight",shape=[in_channel, out_channel],
                                 dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(scope+"bias", [out_channel],initializer=tf.constant_initializer(0.0))
        relu = tf.nn.relu(tf.matmul(input_tensor, weights))
        parameter_list += [weights, biases]
        return relu



#定义网络结构
def build_model(image_size, num_label):
    input_x = tf.placeholder(tf.float32, [None, image_size, image_size, 2], name='input_x')
    label_y = tf.placeholder(tf.float32, [num_label], name='label_y')
    keep_prob = tf.placeholder(tf.float32)

    parameter_list = []
    conv1_1 = conv_layer(input_x,name='conv1_1',out_channel=64, parameter_list=parameter_list)
    conv1_2 = conv_layer(conv1_1,name='conv1_2',out_channel=64, parameter_list=parameter_list)
    pool1 = maxpool_layer(conv1_2,name='pool1')

    conv2_1 = conv_layer(pool1,name='conv2_1',out_channel=128, parameter_list=parameter_list)
    conv2_2 = conv_layer(conv2_1,name='conv2_2',out_channel=128, parameter_list=parameter_list)
    pool2 = maxpool_layer(conv2_2, name='pool2')

    conv3_1 = conv_layer(pool2, name='conv3_1', out_channel=256, parameter_list=parameter_list)
    conv3_2 = conv_layer(conv3_1, name='conv3_2', out_channel=256, parameter_list=parameter_list)
    conv3_3 = conv_layer(conv3_2, name='conv3_3', out_channel=256, parameter_list=parameter_list)
    pool3 = maxpool_layer(conv3_3, name='pool3')

    conv4_1 = conv_layer(pool3, name='conv4_1', out_channel=512, parameter_list=parameter_list)
    conv4_2 = conv_layer(conv4_1, name='conv4_2', out_channel=512, parameter_list=parameter_list)
    conv4_3 = conv_layer(conv4_2, name='conv4_3', out_channel=512, parameter_list=parameter_list)
    pool4 = maxpool_layer(conv4_3, name='pool4')

    conv5_1 = conv_layer(pool4, name='conv5_1', out_channel=512, parameter_list=parameter_list)
    conv5_2 = conv_layer(conv5_1, name='conv5_2', out_channel=512, parameter_list=parameter_list)
    conv5_3 = conv_layer(conv5_2, name='conv5_3', out_channel=512, parameter_list=parameter_list)
    pool5 = maxpool_layer(conv5_3, name='pool5')

    pool5_shape = pool5.get_shape()
    flattened_shape = pool5_shape[1].value * pool5_shape[2].value * pool5_shape[3].value
    flatten_input = tf.reshape(pool5,[-1,flattened_shape],name="flatten_input")

    fc6 = fc_layer(flatten_input, name="fc6", out_channel=4096, parameter_list=parameter_list)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')
    fc7 = fc_layer(fc6_drop, name="fc7", out_channel=4096, parameter_list=parameter_list)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
    logits = fc_layer(fc7_drop, name="fc8", out_channel=1000, parameter_list=parameter_list)
    return input_x, label_y, keep_prob, logits, parameter_list

def build_optimizer(logits, label_y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label_y))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(label_y, axis=1)), tf.float32))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    return loss, accuracy, optimizer

def train(train_x=None, train_y=None):
    def batch_generator():
        for i in range(0):
            yield
    image_size = 224  # 输入图像尺寸
    num_epochs = 1
    input_x, label_y, keep_prob, logits, parameter_list = build_model(image_size, 1000)
    loss, accuracy, optimizer = build_optimizer(logits, label_y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            generator = batch_generator()
            for batch_x, batch_y in generator:
                feed_dict = {input_x:batch_x, label_y:batch_y, keep_prob:0.5}
                batch_loss, batch_accuracy, _ = sess.run([loss, accuracy, optimizer], feed_dict)




if __name__ =='__main__':
    input_x, label_y, keep_prob, logits, parameter_list = build_model(224, 1000)
    loss, accuracy, optimizer = build_optimizer(logits, label_y)

    #train()
