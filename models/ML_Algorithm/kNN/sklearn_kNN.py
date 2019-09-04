from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print(iris_X[:3,:])
print(iris_y[:3])

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

kNN = KNeighborsClassifier()
kNN.fit(X_train, y_train)

print(kNN.predict(X_test))
print(y_test)
print(kNN.score(X_test, y_test))