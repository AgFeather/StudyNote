"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list 表示所有的action 集合
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        #建立q-table，横轴表示所有action，纵轴表示所有state，目前state为空
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)# 查看传入的state是否在当前q-table存在
        # action selection
        if np.random.uniform() < self.epsilon: #选择action奖励最高的行为
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # 当一个 state有多个相同的 Q action value时，idmax会永远返回第一个index, 所以我们乱序state_action的位置，防止idmax()取值一直相同
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        '''
        输入当前state，选择的action，该action对应的奖励，以及到达的新位置。
        根据奖励和新位置更新q table中当前state和对应action的值
        '''
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        '''
        检查输入state是否存在于当前的q table中，如果不存在则向q table加入一行表示这个新state
        '''
        if state not in self.q_table.index: #如果当前q-table不存在传入的state
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
