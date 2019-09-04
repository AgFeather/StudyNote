import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt



def classify(inX, dataSet, labels, k):
	dataSetSize = len(dataSet)
	# calculate distance matrix
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5

	sortedDistIndicies = distances.argsort()
	classCount = {}
	# voting with lowest k distances
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items(),
		key=lambda x:x[1], reverse=True)
	return sortedClassCount[0][0]

def file2matrix():
	filename = "datingTestSet.txt"
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = np.zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		#s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
		line = line.strip()
		#使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		#根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
		if listFromLine[-1] == 'didntLike':
			classLabelVector.append(1)
		elif listFromLine[-1] == 'smallDoses':
			classLabelVector.append(2)
		elif listFromLine[-1] == 'largeDoses':
			classLabelVector.append(3)
		index += 1
	return returnMat, classLabelVector

def autoNorm(dataSet):
	minVal = dataSet.min(0)
	maxVal = dataSet.max(0)
	ranges = maxVal - minVal
	normDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVal, (m, 1))
	normDataSet = normDataSet / np.tile(ranges, (m, 1))
	return normDataSet, ranges, minVal

def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingDataLabels = file2matrix()
	normMat, ranges, minVal = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify(normMat[i,:], normMat[numTestVecs:m,:],\
			datingDataLabels[numTestVecs:m], 3)
		print('the classifier came back with: %d, the real answer is: %d'%(classifierResult, datingDataLabels[i]))
		if (classifierResult != datingDataLabels[i]):
			errorCount += 1.0
	print("the total error rate is: %f"%(errorCount/float(numTestVecs)))



def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input(
		'percentage of time spent playing video games?'))
	ffMiles = float(input(
		'frequent flier miles earned per year?'))
	iceCream = float(input(
		'liters of ice cream consumed per year?'))
	
	datingDataMat, datingDataLabels = file2matrix()
	normMat, ranges, minVal = autoNorm(datingDataMat)
	inArr = np.array([ffMiles, percentTats, iceCream])
	classifierResult = classify((inArr-minVal)/ranges, normMat, datingDataLabels, 3)
	print(' you will probably like this person: ', resultList[classifierResult - 1])




from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def sklearn_kNN():
	knn = KNeighborsClassifier()
	returnMat, classLabelVector = file2matrix()
	X_train, X_test, y_train, y_test = train_test_split(returnMat, classLabelVector, test_size=0.3)
	knn.fit(X_train, y_train)
	prediction = knn.score(X_test, y_test)
	print(prediction)



if __name__ == '__main__':
	

	sklearn_kNN()

	# second datingDataSet

	# datingDataMat, datingDataLabel = file2matrix()
	# print(datingDataMat)
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.scatter(
	# 	datingDataMat[:,1], datingDataMat[:,0], 15.0*np.array(datingDataLabel), 15.0*np.array(datingDataLabel))
	# plt.show()

	datingClassTest()
#	classifyPerson()