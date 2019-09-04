#implement PLA Algorithm with python3
import numpy as np
def PLA(dataset):
	w = np.zeros([2])
	learning_rate = 1
	b = 0
	count = 0
	i = 0
	while i<len(dataset):
		data = np.array(dataset[i][0:2])
		label = dataset[i][-1]
		prediction = np.dot(w,data.T)+b
		if label*prediction<=0:
			w = w + learning_rate*label*data
			b = b + learning_rate*label
			i = 0
			continue
		i+=1
	return w, b


#implement PLA Algorithm dual form with python3
def PLA_dual_form():
	dataset = np.array([[3,3],[4,3],[1,1]])
	label = [1,1,-1]
	n = np.zeros([len(dataset)])
	learning_rate = 1
	b = 0

	temp = []
	for i in range(0,len(dataset)):
		for j in range(0,len(dataset)):
			temp.append(np.dot(dataset[i],dataset[j].T))
	gram_matrix = np.array(temp).reshape([len(dataset),len(dataset)])

	i = 0
	while i<len(dataset):
		tempNum = 0
		for j in range(0,len(dataset)):
			tempNum += n[j]*label[j]*gram_matrix[j][i]
		y = label[i]*(tempNum + b)
		if y<=0:
			n[i]+=1
			b = b+label[i]*learning_rate
			i = 0
			continue
		i+=1

	w = np.zeros([len(dataset[0])])
	for i in range(len(dataset)):
	#	print(i,dataset[i],label[i])
		w = w + n[i]*dataset[i]*label[i]

	return w, b




if __name__ == '__main__':
	dataset = [[3,3,1],[4,3,1],[1,1,-1]]
	w, b = PLA(dataset)
	print('PLA origin form  w: {0}  b: {1}'.format(w, b))

	w, b = PLA_dual_form()
	print('PLA dual form  w: {0}  b: {1}'.format(w, b))
