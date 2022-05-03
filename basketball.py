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
from mysklearn.myclassifiers import MyKNeighborsClassifier, MyDummyClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier
import os
import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myeval
import math

importlib.reload(mysklearn.myutils)
importlib.reload(mysklearn.mypytable)
importlib.reload(mysklearn.myclassifiers)
importlib.reload(mysklearn.myevaluation)
from tabulate import tabulate



np.random.seed(0)

fname = os.path.join("input_data", "cbb.csv")
bball_table = MyPyTable()
bball_table.load_from_file(fname)

fname = os.path.join("input_data", "cbb2022.csv")
bball_table_test = MyPyTable()
bball_table_test.load_from_file(fname)
# stats_header = ['ADJOD','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD',\
    # 'ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O','3P_D','ADJ_T','WAB']

stats_header = ['ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD',\
    'DRB','FTR','3P_O','3P_D','ADJ_T','WAB']

stats_cols = []
stats_cols_inner = []
stats_col = []
for stat in stats_header:
    stats_col.append(myutils.discretize(myutils.normalize(bball_table_test.get_column(stat))))
stats_col.append(bball_table_test.get_column('SEED'))

for index in range(len(bball_table_test.data)):
    for stat_col in stats_col:
        stats_cols_inner.append(stat_col[index])
    stats_cols.append(stats_cols_inner)
    stats_cols_inner = []

X_test = stats_cols.copy()
stats_cols = []
stats_col = []
# Grabbing all the rows we want to use
for stat in stats_header:
    stats_col.append(myutils.discretize(myutils.normalize(bball_table.get_column(stat))))
stats_col.append(bball_table.get_column('SEED'))

# Creating a new table with the rows based on the appropriate columns
for index in range(len(bball_table.data)):
    for stat_col in stats_col:
        stats_cols_inner.append(stat_col[index])
    stats_cols.append(stats_cols_inner)
    stats_cols_inner = []

y_train_bball = [val for val in bball_table.get_column('POSTSEASON')]
X_train_bball = stats_cols.copy()
myForest = MyRandomForestClassifier()
count = 0
myForest.fit(X_train_bball, myutils.discretizeY(y_train_bball), 500, 20, 6, 4)
predictions = myForest.predict()
for i in range(len(predictions)):
    if predictions[i] == myForest.y_test[i]:
        count += 1
print("accuracy:", (count/len(predictions)))

# Step 1: Discretize all of the stats columns into bins
# Step 2: Randomly generate 1/3 of data into a test set and 2/3 into a training set
# X_train_bball, X_test_bball, y_train_bball, y_test_bball = myeval.train_test_split(X_train_bball, y_train_bball, test_size=0.33,random_state=10)
# Step 4: Utilize bootstrap method to generate a forest of possible trees for training set
n = 50
# tree_ratings = []
# my_tree = MyDecisionTreeClassifier()
# for i in range(n):
#     X_sample, X_out_of_bag, y_sample, y_out_of_bag = myeval.bootstrap_sample(X_train_bball, y_train_bball, n_samples=myutils.normal_round(len(X_train_bball) / 2), random_state=i)
#     my_tree.fit(X_sample, y_sample)
#     tree_predictions = my_tree.predict(X_out_of_bag)
#     tree_ratings.append([i , myeval.accuracy_score(y_out_of_bag, tree_predictions)])
#     # TODO: Do we need to randomly create an assortment of available attributes for each of these trees??
#     # TODO: Add more instances to the dataset to meet the 1000 requirement
#     # TODO: use this link: https://barttorvik.com/trank.php?year=2022&sort=&top=0&conlimit=All# 

# for tree_rating in tree_ratings:
#     if tree_rating[1] > 0.5:
#         print(tree_rating)

# # Step 5: 

# X_train_practice = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]]
# y_train_practice = [0,0,0,0,0,1,1,1,1,1]
# bootstrap_solution = myeval.bootstrap_sample(X_train_practice,y_train_practice)
# # print(bootstrap_solution)
# my_tree = MyDecisionTreeClassifier()
# my_tree.fit(X_train_bball,y_train_bball)
# print(my_tree.predict(X_test))