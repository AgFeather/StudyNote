import treePlotter
import basic_DecisionTree

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = basic_DecisionTree.createTree(lenses, lensesLabels)
print(lensesTree)

treePlotter.createPlot(lensesTree)