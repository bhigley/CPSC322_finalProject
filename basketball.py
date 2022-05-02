# some useful mysklearn package import statements and reloads
import importlib

import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils

import mysklearn.mypytable
importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable 
import numpy as np
import mysklearn.myclassifiers
importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier
import os
import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation
import math

importlib.reload(mysklearn.myutils)
importlib.reload(mysklearn.mypytable)
importlib.reload(mysklearn.myclassifiers)
importlib.reload(mysklearn.myevaluation)
from tabulate import tabulate

fname = os.path.join("input_data", "cbb.csv")
bball_table = MyPyTable()
bball_table.load_from_file(fname)

fname = os.path.join("input_data", "cbb2022.csv")
bball_table_test = MyPyTable()
bball_table_test.load_from_file(fname)
# stats_header = ['ADJOD','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD',\
    # 'ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O','3P_D','ADJ_T','WAB']

stats_header = ['ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD',\
    'DRB','FTR','3P_O','3P_D','ADJ_T','WAB','BARTHAG','ADJOE','ADJDE','EFG_O','EFG_D','BARTHAG']#,'SEED']

stats_cols = []
stats_cols_inner = []
stats_col = []
for stat in stats_header:
    stats_col.append(myutils.discretize(myutils.normalize(bball_table_test.get_column(stat))))
stats_col.append(bball_table_test.get_column("SEED"))

for index in range(len(bball_table_test.data)):
    for stat_col in stats_col:
        stats_cols_inner.append(stat_col[index])
    stats_cols.append(stats_cols_inner)
    stats_cols_inner = []

X_test = stats_cols
stats_cols = []
stats_col = []
for stat in stats_header:
    stats_col.append(bball_table.get_column(stat))

for index in range(len(bball_table.data)):
    for stat_col in stats_col:
        stats_cols_inner.append(stat_col[index])
    stats_cols.append(stats_cols_inner)
    stats_cols_inner = []

X_train_bball_origin = [stats for stats in stats_cols]
y_train_bball_origin = [val for val in bball_table.get_column('POSTSEASON')]
# y_train_bball__origin = myutils.discretizeY(y_train_bball_origin)

K_VALUE = 10
print('===========================================')
print("Predictive Accuracy")
print('===========================================')

X_train_folds, X_test_folds = myevaluation.stratified_kfold_cross_validation(X_train_bball_origin,y_train_bball_origin,n_splits=K_VALUE,shuffle=True)
X_train_bball, X_test_bball = [], []
y_dummy_predictions_folds, y_knn_predictions_folds, y_nb_predictions_folds, y_tree_predictions_folds = [], [], [], []
y_actuals_folds = []
y_train_bball,y_test_bball = [], []
for i, X_train_fold in enumerate(X_train_folds):
    for instance_index in X_train_fold:
        X_train_bball.append(X_train_bball_origin[instance_index])
        y_train_bball.append(y_train_bball_origin[instance_index])
    for instance_index in X_test_folds[i]:
        X_test_bball.append(X_train_bball_origin[instance_index])
        y_test_bball.append(y_train_bball_origin[instance_index])
    knn_classifier = MyKNeighborsClassifier(K_VALUE)
    y_actuals_folds.append(y_test_bball)
    knn_classifier.fit(X_train_bball,y_train_bball)
    y_predictions = knn_classifier.predict(X_test_bball)
    y_knn_predictions_folds.append(y_predictions)
    dummy_classifier = MyDummyClassifier()
    dummy_classifier.fit(X_train_bball,y_train_bball)
    y_predictions = dummy_classifier.predict(X_test_bball)
    y_dummy_predictions_folds.append(y_predictions)
    nb_classifier = MyNaiveBayesClassifier()
    nb_classifier.fit(X_train_bball,y_train_bball)
    y_predictions = nb_classifier.predict(X_test_bball)
    y_nb_predictions_folds.append(y_predictions)
    tree_classifier = MyDecisionTreeClassifier()
    tree_classifier.fit(X_train_bball,y_train_bball)
    y_predictions = tree_classifier.predict(X_test_bball)
    y_tree_predictions_folds.append(y_predictions)
    X_train_bball, X_test_bball, y_train_bball, y_test_bball = [], [], [], []
# convert a nested list into a flat list
y_actuals_folds = [item for sublist in y_actuals_folds for item in sublist]
y_knn_predictions_folds = [item for sublist in y_knn_predictions_folds for item in sublist]
y_dummy_predictions_folds = [item for sublist in y_dummy_predictions_folds for item in sublist]
y_nb_predictions_folds = [item for sublist in y_nb_predictions_folds for item in sublist]
y_tree_predictions_folds = [item for sublist in y_tree_predictions_folds for item in sublist]
knn_accuracy = myevaluation.accuracy_score(y_actuals_folds,y_knn_predictions_folds,normalize=True)
# knn_precision = myevaluation.binary_precision_score(y_actuals_folds,y_knn_predictions_folds,labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"],pos_label="yes")
# knn_recall = myevaluation.binary_recall_score(y_actuals_folds,y_knn_predictions_folds,labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"],pos_label="yes")
dummy_accuracy = myevaluation.accuracy_score(y_actuals_folds,y_dummy_predictions_folds,normalize=True)
# dummy_precision = myevaluation.binary_precision_score(y_actuals_folds,y_dummy_predictions_folds,labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"],pos_label="yes")
# dummy_recall = myevaluation.binary_recall_score(y_actuals_folds,y_dummy_predictions_folds,labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"],pos_label="yes")
nb_accuracy = myevaluation.accuracy_score(y_actuals_folds,y_nb_predictions_folds,normalize=True)
# nb_precision = myevaluation.binary_precision_score(y_actuals_folds,y_nb_predictions_folds,labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"],pos_label="yes")
# nb_recall = myevaluation.binary_recall_score(y_actuals_folds,y_nb_predictions_folds,labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"],pos_label="yes")
tree_accuracy = myevaluation.accuracy_score(y_actuals_folds,y_tree_predictions_folds,normalize=True)
# tree_precision = myevaluation.binary_precision_score(y_actuals_folds,y_tree_predictions_folds,labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"],pos_label="yes")
# tree_recall = myevaluation.binary_recall_score(y_actuals_folds,y_tree_predictions_folds,labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"],pos_label="yes")
print("Stratified 10-Fold Cross Validation")
print("k Nearest Neighbors Classifier: ")
print("accuracy =",knn_accuracy,"error rate =",1 - knn_accuracy)
# print("precision =",knn_precision,"recall =",knn_recall)
# matrix = myevaluation.confusion_matrix(y_actuals_folds, y_knn_predictions_folds, labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"])
# print(tabulate(matrix,headers=["Champions","2ND","F4","E8","S16","R32","R64","R68"]))
print("Dummy Classifier: ")
print("accuracy =",dummy_accuracy,"error rate = ",1 - dummy_accuracy)
# print("precision =",dummy_precision,"recall =",dummy_recall)
# matrix = myevaluation.confusion_matrix(y_actuals_folds, y_dummy_predictions_folds, labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"])
# print(tabulate(matrix,headers=["Champions","2ND","F4","E8","S16","R32","R64","R68"]))
print("Naive Bayes Classifier: ")
print("accuracy =",nb_accuracy,"error rate = ",1 - nb_accuracy)
# print("precision =",nb_precision,"recall =",nb_recall)
# matrix = myevaluation.confusion_matrix(y_actuals_folds, y_nb_predictions_folds, labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"])
# print(tabulate(matrix,headers=["Champions","2ND","F4","E8","S16","R32","R64","R68"]))
print("Decision Tree Classifier: ")
print("accuracy =",tree_accuracy,"error rate = ",1 - tree_accuracy)
# print("precision =",tree_precision,"recall =",tree_recall)
# matrix = myevaluation.confusion_matrix(y_actuals_folds, y_tree_predictions_folds, labels=["Champions","2ND","F4","E8","S16","R32","R64","R68"])
