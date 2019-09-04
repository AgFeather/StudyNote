import matplotlib.pyplot as plt
import numpy as np



def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()
	m, n = np.shape(dataMatrix)
	alpha = 0.01
	maxCycles = 500
	weights = np.ones((n, 1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights


def plotBestFit(wei):
#	weights = wei.getA()
	dataMat, labelMat = loadDataSet()
	dataArr = np.array(dataMat)
	n = np.shape(dataArr)[0]
	xcord1 = []
	ycord1 = []
	xcord0 = []
	ycord0 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i, 1])
			ycord1.append(dataArr[i, 2])
		else:
			xcord0.append(dataArr[i, 1])
			ycord0.append(dataArr[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord0, ycord0, s=30, c='red', marker='s')
	ax.scatter(xcord1, ycord1, s=30, c='green')
	x = np.arange(-3.0, 3.0, 0.1)
	y = (-weights[0]-weights[1]*x) / weights[2]
	ax.plot(x, y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()


def stocGradAscent(dataMatrix, classLabels, numIter=150):
	m,n = np.shape(dataMatrix)												#返回dataMatrix的大小。m为行数,n为列数。
	weights = np.ones(n)   													#参数初始化
	weights_array = np.array([])											#存储每次更新的回归系数
	for j in range(numIter):											
		dataIndex = list(range(m))
		for i in range(m):			
			alpha = 4/(1.0+j+i)+0.01   	 									#降低alpha的大小，每次减小1/(j+i)。
			randIndex = int(np.random.uniform(0,len(dataIndex)))				#随机选取样本
			h = sigmoid(sum(dataMatrix[randIndex]*weights))					#选择随机选取的一个样本，计算h
			error = classLabels[randIndex] - h 								#计算误差
			weights = weights + alpha * error * dataMatrix[randIndex]   	#更新回归系数
			weights_array = np.append(weights_array,weights,axis=0) 		#添加回归系数到数组中
			del(dataIndex[randIndex]) 										#删除已经使用的样本
	weights_array = weights_array.reshape(numIter*m,n) 						#改变维度
	return weights,weights_array 		
			#f'(x) = f(x) * (1 - f(x))



if __name__ == '__main__':
	dataArr, labelMat = loadDataSet()
	weights1 = gradAscent(np.array(dataArr), labelMat)
	weights, weights_array= stocGradAscent(np.array(dataArr), labelMat)
	print(weights1)
	print(weights)

	plotBestFit(weights1)
	