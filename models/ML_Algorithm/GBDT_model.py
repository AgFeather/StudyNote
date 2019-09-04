import progressbar
from utils.loss_function import SquareLoss, SoftmaxLoss
import numpy as np
from decision_tree import RegressionTree


bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]




class GBDT(object):
    def __init__(self,
                 n_estimators, # int 树的个数
                 learning_rate, # 梯度下降的学习速率
                 min_samples_split, # 每棵子树叶子节点中数据的最小数目
                 min_impurity, # 每棵子树的最小纯度
                 max_depth, # 每棵子树的最大深度
                 regression=True # boolean 是否为回归问题
                 ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        if regression:
            self.loss = SquareLoss()
        else:
            self.loss = SoftmaxLoss()

        self.trees = []
        for _ in range(self.n_estimators):
            tree = RegressionTree(min_samples_split=self.min_samples_split,
                                  min_impurity=self.min_impurity,
                                  max_depth=self.max_depth)

    def fit(self, x, y):
        self.trees[0].fit(x, y)
        y_pred = self.trees[0].predict(x)
        for i in self.bar(range(1, self.n_estimators)):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(x, gradient)
            y_pred += np.multiply(self.learning_rate, self.trees[i].predict(x))

    def predict(self, x):
        y_pred = self.trees[0].predict[x]
        for i in self.bar(range(1, self.n_estimators)):
            y_pred += np.multiply(self.learning_rate, self.trees[i].predict(x))

        if not self.regression:
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            y_pred = np.argmax(y_pred, axis=1)

        return y_pred


class GBDTRegressor(GBDT):
    def __init__(self,
                 n_estimators=100, # int 树的个数
                 learning_rate=0.2, # 梯度下降的学习速率
                 min_samples_split=2, # 每棵子树叶子节点中数据的最小数目
                 min_var_red=1e-7, # 每棵子树的最小纯度
                 max_depth=4, # 每棵子树的最大深度
                 ):
        super(GBDTRegressor, self).__init__(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            min_samples_split=min_samples_split,
                                            min_impurity=min_var_red,
                                            max_depth=max_depth,
                                            regression=True)
