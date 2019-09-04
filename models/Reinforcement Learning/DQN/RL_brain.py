import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# deep q network off-policy
class DeepQNetWork(object):
    def __init__(self,
            n_actions, # action个数
            n_features, # 多少个observation
            learning_rate=0.01,
            reward_decay=0.9, # reward衰减率
            e_greedy=0.9, # 90%概率选择 贪婪选择Q最大的行为
            replace_target_iter=300, # 每隔多少步更新target net的参数
            memory_size=500, # 记忆库容量大小
            batch_size=32,
            e_greedy_increment=None, # 不断缩小greedy的范围
            output_graph=False,):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_] 创建记忆库：n_features*2+2 = observation+observation_+action+reward
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = [] # 记录每一步的误差


    def _build_net(self):
        '''
        建立两个结构相同但参数不同的NN eval_net 和 target_net,
        其中target_net是不训练的，每隔指定步数就将eval_net的参数更新到target_net中
        '''
        # build evaluate net
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s') #输入state，nn预测出针对当前state执行各个动作的reward
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target') # 真是环境中对各个动作的reward
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1) #config of layers

            with tf.variable_scope('l1'):
                # 第一层NN
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer,collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                # 第二层NN
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        # build target net
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_') #s_ 表示下一个state
        with tf.variable_scope('target_net'):
            #c_names(collection names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'): # hasttr()判断对象是否包含对应的属性
            self.memory_counter = 0 # 从记忆库表的第零行开始
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new  memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # 将observation输入NN，得到各个action的预测评分，选择最大评分的action
        observation = observation[np.newaxis, :] # to feed the shape of input layer

        if np.random.uniform() < self.epsilon:
            # 将当前state的observation传入NN，进行正向传播得到各个action的prediction reward
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s:observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action



    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            #将eval net的参数更新到target net中
            self.sess.run(self.replace_target_op)
            print('\n target_params_replaced \n')

        # 从记忆库随机sample采样batch_size个记忆条用于训练
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else: # 初始状态，记忆库还很小，所以从已经更新的memory_counter中random sample
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # q_eval 和 q_next 包含所有action对应的奖励估计值
        q_eval, q_next = self.sess.run([self.q_eval, self.q_next],
            feed_dict={
                self.s:batch_memory[:, :self.n_features],
                self.s_:batch_memory[:, -self.n_features:], # memory后n_feature是next state
        })

        # 特殊的地方：在eval_net对所有action预测reward后，其实我们只选择一个action进行执行，其他的action的reward并不需要
        # 所以在进行反向传播时，我们应该只使用选中的action对应的reward计算的loss更新网络，其余action的差值应该置于0
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int) # 取记忆库中所有action(对应的index正好为n_feature)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # 将这(q_target - q_eval) 当成误差, 反向传递会神经网络.
        # 所有为 0 的 action 值是没有被选择的 action, 之前有选择的 action 才有不为0的值.
        # 我们只反向传递之前选择的 action 的值,

        # 训练eval net
        _, self.cost = self.sess.run([self._train_op, self.loss],
            feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target:q_target})

        self.cost_his.append(self.cost) # 记录cost误差

        # 逐渐增加epsilon，降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('cost')
        plt.xlabel('training steps')
        plt.show()
        '''
        可以看出曲线并不是平滑下降的, 这是因为 DQN 中的 input 数据是一步步改变的,
        而且会根据学习情况, 获取到不同的数据. 所以这并不像一般的监督学习, DQN 的 cost 曲线就有所不同了.

        '''
