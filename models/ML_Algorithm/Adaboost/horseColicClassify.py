import model

def loadDataSet(fileName):
    dataMat = []
    labelMat= []
    fr = open(fileName， 'r')
    numFeat = len(fr.readline().split('\t'))
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine(i)))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = model.adaBoostTrainDS(dataArr, labelArr, 10)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction = model.adaClassify(testArr, classifierArray)
    errArr = np.mat(np.ones((67, 1)))
    errArr[prediction != np.mat(testLabelArr).T].sum()
