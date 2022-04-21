from multiprocessing import dummy
import myutils_ben as myutils
from mypytable_ben import MyPyTable
from myclassifiers_ben import MyDecisionTreeClassifier, MyDummyClassifier
import numpy as np

bball_table = MyPyTable()
bball_table.load_from_file("input_data/cbb.csv")

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
test_indexes = [np.random.randint(0,240) for i in range(25)]
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
print(predictions_dummy)
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