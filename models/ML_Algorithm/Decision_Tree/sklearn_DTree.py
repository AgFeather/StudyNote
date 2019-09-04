from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_data = iris.data
y_labels = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.3)

DTree = DecisionTreeClassifier()
DTree.fit(X_train, y_train)
prediction = DTree.predict(X_test)
print(prediction)
print(y_test)
print(DTree.score(X_test, y_test))
