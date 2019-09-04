import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATEE = 6 # the length of 1 dimensional world
ACTIONS = ['left', 'right'] #available actions
EPSILON = 0.9 #greedy police
ALPHA = 0.1 # learning rate
LAMBDA = 0.9 # discount factor 未来奖励衰减值
MAX_EPISODES = 7 # maximum episodes
FRESH_TIME = 0.3 # fresh time for one move


def build_q_table(n_states, actions):
    '''
    创建q表，q表是一个二维表，index对应所有state，column对应所有action
    '''
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )
    # print(table)
    return table

def choose_action(state, q_table):
    '''
    定义探索者是如何挑选行为的. 引入 epsilon greedy 的概念.
    因为在初始阶段, 随机的探索环境, 往往比固定的行为模式要好, 所以这也是累积经验的阶段, 我们希望探索者不会那么贪婪.
    EPSILON 就是用来控制贪婪程度的值. EPSILON 可以随着探索时间不断提升(越来越贪婪),
    不过在这个例子中, 我们就固定成 EPSILON = 0.9, 90% 的时间是选择最优策略, 10% 的时间来探索.
    '''
    # this how to choose an action
    state_actions = q_table.iloc[state, :]  #去的当前state所有action的值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        ## 非贪婪 or 或者这个 state 还没有探索过，就随机选择一个action进行行动
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax() #贪婪模式，选择值最大的action作为行动
        #action_name = state_actions.idmax()
    return action_name

def get_env_feedback(S, A):
    '''
    做出行为后, 环境也要给我们的行为一个反馈, 反馈出下个 state (S_)
    和 在上个 state (S) 做出 action (A) 所得到的 reward (R).
    这里定义的规则就是, 只有当 o 移动到了 T, 探索者才会得到唯一的一个奖励, 奖励值 R=1, 其他情况都没有奖励.

    S: 当前的state   A：采取的行动
    return S_：在S采取行动A后到达的state   return R：到达S_后获得的奖励
    '''
    if A == 'right': #move right
        if S == N_STATEE - 1 - 1: # terminate到达终点，获得奖励
            # 第一个减一因为N_STATE最后一位是奖励，第二个减一是因为用的是S进行比较，
            # 而我们真正需要比较的事S_ = S + 1时的奖励状态
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else: #move left
        R = 0
        if S == 0:
            S_ = S # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    '''
    更新环境
    '''
    env_list = ['-'] * (N_STATEE-1) + ['T'] # '-------T'，表示环境
    if S == 'terminal': # 已经到达终点获得奖励
        print('\rEpisode %s: total_steps = %s'%(episode+1, step_counter))
        time.sleep(2)
        print('\r          ')
    else:
        env_list[S] = 'o'
        # interaction = ''.join(env_list)
        # print('\r{}'.format(interaction))
        print('\r', env_list)
        time.sleep(FRESH_TIME)


def RL():
    '''
    强化学习核心模块，不断循环更新q表和环境
    '''
    q_table = build_q_table(N_STATEE, ACTIONS)
    for episode in range(MAX_EPISODES): #训练epoch
        step_counter = 0 #该轮训练中迭代次数
        S = 0 # 初始state
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated: # 在仍旧没有到达终点时循环
            A = choose_action(S, q_table) # 选择行为
            S_, R = get_env_feedback(S, A)# 实施行为并得到环境的反馈：下一个个state和对应获得的奖励
            q_predict = q_table.ix[S, A] # 找到当前[S, A]处的q值，进行更新
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()#使用新state所有可能的奖励的最大q值更新上一个state的q值
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)# 根据reward，更新[S,A]处更Q值
            S = S_
            update_env(S, episode, step_counter+1) #update environment
            step_counter += 1

    return q_table


if __name__ == '__main__':
    q_table = RL()
    print(q_table)
