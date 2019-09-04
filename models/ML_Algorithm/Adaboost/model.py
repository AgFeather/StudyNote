import numpy as np

def loadSimpData():
    datMat = np.matrix([[ 1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对数据进行分类，所有在阈值一遍的数据会被分到类别-1， 另一边的会被分到+1
    '''
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = 1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    '''
    针对给定数据集，遍历每个特征的每个按步长取值，分别构建分类树墩，
    并返回最佳树墩（加权error最小），其中参数D为数据集中每个实例的权重list
    '''
    dataMatrix = np.mat(dataArr);
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf  #将最小误差率初始化为正无穷
    for i in range(n):#对于数据中的每一个特征
        rangeMin = dataMatrix[:, i].min()#找到该特征的最小值
        rangeMax = dataMatrix[:, i].max()#找到该特征的最大值
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):#对于每一个步长（特征取值步长）
            for inequal in ['lt', 'gt']:#对于每一个不等号
                threshVal = (rangeMin + float(j) * stepSize)
                #根据当前的threshold值，构建一个单层决策树墩，并返回对数据的预测分类
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0#计算误分类个数
                weightedError = D.T * errArr#计算加权误分类的和（D表示权重）
                if weightedError < minError:
                    minError = weightedError#更新最小加权error
                    bestClasEst = predictedVals.copy()
                    #记录最佳分类树墩的特征维度i，threshold值，不等号方式
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst#返回树墩字典，最小加权误差，最佳树墩预测结果

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = [] #存放每个决策树墩的信息
    m = np.shape(dataArr)[0]#数据个数
    D = np.mat(np.ones((m, 1))/m)#每个数据的权重
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)#取得最佳树墩
        print("D: ", D.T)
        alpha = float(0.5 * np.log((1.0-error)/max(error, 1e-6)))#计算树墩的权重alpha
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst: ', classEst.T)#打印预测结果
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)#进行数据权重更新
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst#计算整个模型累加到当前时对数据的预测分类值。
        print('aggClassEst: ', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) !=
                        np.mat(classLabels).T, np.ones((m, 1)))#计算错误率累加
        errorRate = aggErrors.sum()/m
        print('Total Error: ', errorRate)
        if errorRate == 0.0: break
    return weakClassArr

#对训练好的Adaboost进行测试
def adaClassify(dataToClass, classifierArr):
    print('Test adaboost model...')
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                    classifierArr[i]['thresh'],
                    classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    print(np.sign(aggClassEst))
    return np.sign(aggClassEst)

if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    D = np.mat(np.ones((5, 1))/5)
    #bestStump, minError, bestClasEst = buildStump(dataMat, classLabels, D)
    #training adaboost
    classifierArray = adaBoostTrainDS(dataMat, classLabels)
    #testing adaboost
    adaClassify([[0, 0], [5, 5]], classifierArray)
    #可以发现，随着迭代的进行，Adaboost对模型的分类结果越来越强
