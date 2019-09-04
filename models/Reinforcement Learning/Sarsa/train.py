from maze_env import Maze
from RL_brain import SarsaTable

def update():
    #学习100遍
    for episode in range(100):
        #初始化state的观测值
        observation = env.reset()
        # Sarsa根据state的观测值挑选action, 和q learning不同，Sarsa首先是在外循环获得action
        action = RL.choose_action(str(observation))

        while True:
            #更新可视化环境
            env.render()

            # 在环境中采取行为，获得下一个state_，reward，和是否终止
            observation_, reward, done = env.step(action)

            # 根据下一个state_ 选取下一个action_(于q learning不同的地方，Sarsa一定会执行选取的行为)
            action_ = RL.choose_action(str(observation_))

            # 从 (s, a, r, s, a) 中学习, 更新 Q_tabel 的参数 ==> Sarsa。 学习中药考虑下一个action_,而q learning会选择max
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个当成下一步的 state (observation) and action
            observation = observation_
            action = action_ # 表示下一个回合一定采取这个回合输出的action

            # 如果掉下地狱或者升上天堂, 这回合就结束了
            if done:
                break

    # 结束游戏并关闭窗口
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # 定义环境 env 和 RL 方式
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    # 开始可视化环境 env
    env.after(100, update)
    env.mainloop()
