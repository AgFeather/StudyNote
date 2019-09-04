# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
import numpy as np
import random


def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
	m,n = np.shape(dataMatrix)												#返回dataMatrix的大小。m为行数,n为列数。
	weights = np.ones(n)   													#参数初始化										#存储每次更新的回归系数
	for j in range(numIter):											
		dataIndex = list(range(m))
		for i in range(m):			
			alpha = 4/(1.0+j+i)+0.01   	 									#降低alpha的大小，每次减小1/(j+i)。
			randIndex = int(random.uniform(0,len(dataIndex)))				#随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights))					#选择随机选取的一个样本，计算h
			error = classLabels[randIndex] - h 								#计算误差
			weights = weights + alpha * error * dataMatrix[randIndex]   	#更新回归系数
			del(dataIndex[randIndex]) 										#删除已经使用的样本
	return weights 															#返回



def gradAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)										#转换成numpy的mat
	labelMat = np.mat(classLabels).transpose()							#转换成numpy的mat,并进行转置
	m, n = np.shape(dataMatrix)											#返回dataMatrix的大小。m为行数,n为列数。
	alpha = 0.01														#移动步长,也就是学习速率,控制更新的幅度。
	maxCycles = 500														#最大迭代次数
	weights = np.ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)								#梯度上升矢量化公式
		error = labelMat - h
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights.getA()												#将矩阵转换为数组，并返回




def colicTest():
	frTrain = open('horseColicTraining.txt')										#打开训练集
	frTest = open('horseColicTest.txt')												#打开测试集
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels,500)		#使用改进的随即上升梯度训练
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec) * 100 								#错误率计算
	print("测试集错误率为: %.2f%%" % errorRate)
	return errorCount

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0


def colicSklearn():
	frTrain = open('horseColicTraining.txt')										#打开训练集
	frTest = open('horseColicTest.txt')												#打开测试集
	trainingSet = []; trainingLabels = []
	testSet = []; testLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[-1]))
	for line in frTest.readlines():
		currLine = line.strip().split('\t')
		lineArr =[]
		for i in range(len(currLine)-1):
			lineArr.append(float(currLine[i]))
		testSet.append(lineArr)
		testLabels.append(float(currLine[-1]))
	classifier = LogisticRegression(solver = 'sag',max_iter = 5000).fit(trainingSet, trainingLabels)
	test_accurcy = classifier.score(testSet, testLabels) * 100
	print('正确率:%f%%' % test_accurcy)




def multiTest():
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print('after %d iterations the average error rate is: %f' % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':
	 
	multiTest()
