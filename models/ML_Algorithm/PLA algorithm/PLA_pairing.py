#使用python3实现对偶形式的感知机算法



def PLA():
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
		print(i,dataset[i],label[i])
		w = w + n[i]*dataset[i]*label[i]

	return w, b
