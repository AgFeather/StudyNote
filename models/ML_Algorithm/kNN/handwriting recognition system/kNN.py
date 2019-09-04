import numpy as np
from os import listdir

def kNN_classify(inX, dataSet, labels, k=3):
	x_matrix = np.tile(inX, [dataSet.shape[0], 1])
	distance_matrix = (x_matrix - dataSet) ** 2
	distance_matrix = np.sum(distance_matrix, axis=1)
	distance_matrix = distance_matrix ** 0.5
	sorted_distance = distance_matrix.argsort()
	class_count = {}
	for i in range(0, k):
		class_label = labels[sorted_distance[i]]
		class_count[class_label] = class_count.get(class_label, 0) + 1
	prediction = sorted(class_count.items(), key=lambda x:x[1], reverse=True)
	return prediction[0][0]



def img2vector(filename):
	returnVect = np.zeros([1, 32 * 32])
	file = open(filename)
	for i in range(32):
		lineStr = file.readline()
		for j in range(32):
			returnVect[0, 32 * i + j] = int(lineStr[j])
	return returnVect


def handwriting_class():
	hw_labels = []
	training_file_list = listdir('trainingDigits')
	trainingSet_size = len(training_file_list)
	trainingMat = np.zeros([trainingSet_size, 1024])
	for i in range(trainingSet_size):
		file_name_dir = training_file_list[i]
		fileStr = file_name_dir.split('.')[0]
		class_number = fileStr.split('_')[0]
		hw_labels.append(class_number)
		trainingMat[i] = img2vector('trainingDigits/%s'%file_name_dir)

	test_file_list = listdir('testDigits')
	errorCount = 0
	test_size = len(test_file_list)
	for i in range(test_size):
		file_name_dir = test_file_list[i]
		fileStr = file_name_dir.split('.')[0]
		test_label = fileStr.split('_')[0]
		test_data = img2vector('testDigits/%s'%file_name_dir)
		prediction = kNN_classify(test_data, trainingMat, hw_labels, 5)
		if prediction is not test_label:
			errorCount += 1
		if i % 20 == 0:		
			accuracy = float(1 - errorCount/(i+1))
			print('test label is %s, prediction is %s, accuracy is %.4f'%(test_label, prediction,accuracy))

	print('finish')

if __name__ == '__main__':
	# imgVect = img2vector('./trainingDigits/0_0.txt')
	# print(imgVect[0,0:31])
	handwriting_class()