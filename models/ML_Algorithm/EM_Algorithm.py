'''
用python实现EM算法，解决双硬币问题。

双硬币问题：
假如有两枚不均匀硬币A，B，以相同概率选择其中一枚，进行抛硬币实验，每次一共抛10次。一共选择了5次硬币。实验结果如下：

B: [H T T T H H T H T H]
A: [H H H H T H H H H H]
A: [H T H H H H H T H H]
B: [H T H T T T H H T T]
A: [T H H H T H H H T H]

其中，H表示正面朝上，T表示背面。

现在假设有a，b两种情况：
a表示实习生记录了详细的试验数据，我们可以观测到试验数据中每次选择的是A还是B
b表示实习生忘了记录每次试验选择的是A还是B，我们无法观测实验数据中选择的硬币是哪个
问在两种情况下分别如何估计两个硬币正面出现的概率？
'''

import numpy as np
import math





class EM_Model_v1(object):
    def __init__(self):
        pass

    def run_em(self, observations, prior):
        thetaA, thetaB = prior
        m, n = observations.shape
        headNums = observations.sum(axis=1) #每次实验中正面向上的次数
        tailNums = n - headNums   # 背面向上的次数

        for i in range(20):
            pAHNum, pATNum, pBHNum, pBTNum = 0, 0, 0, 0
            for headNum, tailNum in zip(headNums, tailNums):

                # E步，计算本次抛币实验来自A，B的概率
                pHeadA = math.pow(thetaA, headNum)
                pTailA = math.pow((1-thetaA), (tailNum))
                pA = pHeadA * pTailA  # 本次抛币实验来自于A硬币的概率

                pHeadB = math.pow(thetaB, headNum)
                pTailB = math.pow((1-thetaB), tailNum)
                pB = pHeadB + pTailB # 本次抛币实验来自于B硬币的概率

                # 对两个概率归一化
                pSum = pA + pB
                pA = pA / pSum
                pB = pB / pSum

                # E步，计算抛币实验选择各个硬币且正面朝上次数的期望
                pAHNum += headNum * pA
                pATNum += tailNum * pA
                pBHNum += headNum * pB
                pBTNum += tailNum * pB

            # M步，求解产生E期望的最可能的概率值
            thetaA = pAHNum / (pAHNum + pATNum)
            thetaB = pBHNum / (pBHNum + pBTNum)

        return thetaA, thetaB





from scipy import stats
class EM_Model_v2(object):
    def __init__(self):
        pass

    def single_step(self, priors, observations):
        '''
        priors: [thetaA, thetaB]
        observations: [m * n matrix]
        return: new_priors
        '''
        counts = {'A':{'H':0, 'T':0},
                  'B':{'H':0, 'T':0}}
        thetaA, thetaB = priors

        # E step
        for observation in observations:
            length = len(observation)
            num_heads = np.sum(observation)
            num_tails = length - num_heads
            contributionA = stats.binom.pmf(num_heads, length, thetaA) # A和B的二项分布
            contributionB = stats.binom.pmf(num_heads, length, thetaB)
            weightA = contributionA / (contributionA + contributionB)
            weightB = contributionB / (contributionA + contributionB)

            # 更新在当前参数下A、B硬币产生的正反面次数
            counts['A']['H'] += weightA * num_heads
            counts['A']['T'] += weightA * num_tails
            counts['B']['H'] += weightB * num_heads
            counts['B']['T'] += weightB * num_tails

        # M step
        new_thetaA = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
        new_thetaB = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])

        return [new_thetaA, new_thetaB]


    def run_em(self, observations, prior, tol=1e-6, iterations=1000):
        '''
        tol: 迭代结束阈值
        iterations: 最大迭代次数
        '''
        iteration = 0
        while iteration < iterations:
            new_prior = self.single_step(prior, observations)
            delta_change = np.abs(prior[0] - new_prior[0])
            if delta_change < tol:
                break
            else:
                prior = new_prior
                iteration += 1
        return [new_prior, iteration]

if __name__ == '__main__':

    observations = np.array([[1,0,0,0,1,1,0,1,0,1],
                             [1,1,1,1,0,1,1,1,0,1],
                             [1,0,1,1,1,1,1,0,1,1],
                             [1,0,1,0,0,0,1,1,0,0],
                             [0,1,1,1,0,1,1,1,0,1]]) # 1表示正面朝上

    # 初始化两种硬币正面的概率
    prior = [0.5, 0.6] # thetaA, thetaB

    model1 = EM_Model_v1()
    prior = model1.run_em(observations, prior)
    print(prior)

    model2 = EM_Model_v2()
    prior = model2.run_em(observations, prior)
    print(prior)
