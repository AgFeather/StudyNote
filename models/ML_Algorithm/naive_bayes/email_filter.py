import numpy as np
import re
import basic_bayes

def textParse(bigString):
	listOfTokens = re.split(r'\W*', bigString)
	return [token.lower() for token in listOfTokens if(len(token) > 2)]

def spamTest():
	docList = []
	classList = []
	fullText = []
	for i in range(1, 26):
		wordList = textParse(open('email/spam/%d.txt'%i, 'r').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)

	for i in range(1, 26):
		wordList = textParse(open('email/ham/%d.txt'%i, 'r').read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)

	vocabDict = basic_bayes.createVocabList(docList)
	trainingSet = range(50)
	testSet = []

	for i in range(10):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])

	trainMat = []
	trainClass = []
	for i in trainingSet:
		wordMat = basic_bayes.setOfWords2Vec(vocabDict, docList[i])
		trainMat.append(wordMat)
		trainClass.append(classList[i])

	p0V, p1V, pSpam = basic_bayes.trainNB(np.array(trainMat), np.array(trainClass))
	errorCount = 0
	for i in testSet:
		TestWordVec = basic_bayes.setOfWords2Vec(vocabDict, docList[i])
		prediction = basic_bayes.classifyNB(np.array(TestWordVec), p0V, p1V, pSpam)
		if prediction != classList[i]:
			errorCount += 1
	print('the error rate is:', float(errorCount) / len(testSet))


if __name__ == '__main__':
	spamTest()