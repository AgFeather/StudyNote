import numpy as np




def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', \
		'problems', 'help', 'please'],
		['maybe', 'not', 'take', 'him', \
		'to', 'dog', 'park', 'stupid'],
		['my', 'dalmation', 'is', 'so', 'cute', \
		'I', 'love', 'him'],
		['stop', 'posting', 'stupid', 'worthless', 'garbage'],
		['mr', 'licks', 'ate', 'my', 'steak', 'how',\
		'to', 'stop', 'him'],
		['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]
	#1 is abusive, 0 not
	return postingList,classVec


def createVocabList(dataSet): # create a wordSet with all the words in dataSet
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		else:
			print('this word is not in my vocabulary')
	return returnVec


def trainNB(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory) / float(numTrainDocs)
	p0Num = np.ones(numWords) #lapulasi pinghua
	p1Num = np.ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = np.log(p1Num/p1Denom)
	p0Vect = np.log(p0Num/p0Denom)
	return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
	p1 = np.sum(vec2Classify * p1Vect) + np.log(pClass1)
	p0 = np.sum(vec2Classify * p0Vect) + np.log(1 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	dataSet, labels = loadDataSet()
	VocabBags = createVocabList(dataSet)
	trainMat = []
	for data in dataSet:
		trainMat.append(setOfWords2Vec(VocabBags, data))
	p0Vect, p1Vect, pAbusive = trainNB(trainMat, labels)
	testEntry = ['love', 'my', 'dalmation']
	testVec = np.array(setOfWords2Vec(VocabBags, testEntry))
	prediction = classifyNB(testVec, p0Vect, p1Vect, pAbusive)
	print(testEntry, 'is classified as: ', prediction)

if __name__ == '__main__':
	testingNB()
	