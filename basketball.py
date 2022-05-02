# some useful mysklearn package import statements and reloads
import importlib

import mysklearn.myutils
# importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils

import mysklearn.mypytable
# importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable 
import numpy as np
import mysklearn.myclassifiers
# importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import os
import mysklearn.myevaluation
# importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation
import math

# importlib.reload(mysklearn.myutils)
# importlib.reload(mysklearn.mypytable)
# importlib.reload(mysklearn.myclassifiers)
# importlib.reload(mysklearn.myevaluation)
from tabulate import tabulate

# fname = os.path.join("input_data", "cbb.csv")
# bball_table = MyPyTable()
# bball_table.load_from_file(fname)

# fname = os.path.join("input_data", "cbb2022.csv")
# bball_table_test = MyPyTable()
# bball_table_test.load_from_file(fname)
# # stats_header = ['ADJOD','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD',\
#     # 'ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O','3P_D','ADJ_T','WAB']

# stats_header = ['ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD',\
#     'DRB','FTR','3P_O','3P_D','ADJ_T','BARTHAG','ADJOE','ADJDE','EFG_O','EFG_D','BARTHAG']#,'WAB']#,'SEED']

# stats_cols = []
# stats_cols_inner = []
# stats_col = []
# for stat in stats_header:
#     stats_col.append(myutils.discretize(myutils.normalize(bball_table_test.get_column(stat))))
# stats_col.append(bball_table_test.get_column("SEED"))

# for index in range(len(bball_table_test.data)):
#     for stat_col in stats_col:
#         stats_cols_inner.append(stat_col[index])
#     stats_cols.append(stats_cols_inner)
#     stats_cols_inner = []

# X_test = stats_cols
# stats_cols = []
# stats_col = []
# for stat in stats_header:
#     stats_col.append(bball_table.get_column(stat))

# for index in range(len(bball_table.data)):
#     for stat_col in stats_col:
#         stats_cols_inner.append(stat_col[index])
#     stats_cols.append(stats_cols_inner)
#     stats_cols_inner = []

# X_train_bball_origin = [stats for stats in stats_cols]
# y_train_bball_origin = [val for val in bball_table.get_column('POSTSEASON')]

bball_table = MyPyTable()
bball_table.load_from_file("input_data/cbb.csv")

myutils.normalize(bball_table.data)

X = []
y = []
X_train = []
X_test = []
y_train = []
y_test = []
sum_accuracy = 0
k = 10
sum_accuracy_copy = 0

# test_indexes = [(2*i) for i in range(20)]
test_indexes = [np.random.randint(0,200) for i in range(25)] # R64 starts at table index 170
print(test_indexes)

j = 0
for row in bball_table.data:
    x_row = []
    x_row.append(row[-1]) # tournament seed
    x_row.append(row[-2])
    if j in test_indexes:
        X_test.append(x_row)
        y_test.append(row[21])
    else:
        y.append(row[21]) # which team won
        X.append(x_row)
    j += 1


# print(X_test, y_test)
treeClassifier = MyDecisionTreeClassifier()
treeClassifier.fit(X, y)
dummyClassifier = MyDummyClassifier()
dummyClassifier.fit(X, y)
predictions_dummy = dummyClassifier.predict(X_test)
predictions = treeClassifier.predict(X_test)

j = 0
correct = 0
# print(predictions_dummy)
print(predictions)
print(y_test)

for val in predictions:
    if val == y_test[j]:
        correct += 1
    j += 1
accuracy = correct / len(y_test)
print("accuracy", accuracy)
print("error", 1 - accuracy)
# print(myevaluation.binary_precision_score(y_test, predictions))
# print(myevaluation.binary_recall_score(y_test, predictions))
j = 0
correct = 0
for item in predictions_dummy:
    if item == y_test[j]:
        correct += 1
    j += 1
accuracy = correct / len(y_test)
print("accuracy", accuracy)
print("error", 1 - accuracy)
# print(myevaluation.binary_precision_score(y_test, predictions_dummy))
# print(myevaluation.binary_recall_score(y_test, predictions_dummy))


i = 0