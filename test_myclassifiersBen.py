import numpy as np
from sklearn import tree

from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier

def test_random_forest_classifier_fit():
    myForest = MyRandomForestClassifier()

    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    
    myForest.fit(X_train_interview, y_train_interview, 9, 3, 2, random_state=0)
    print(myForest.predict())
    assert False is True

def test_random_forest_classifier_predict():
    assert False is True

# def test_decision_tree_classifier_fit():
#     treeClassifier = MyDecisionTreeClassifier()

#     X_train_interview = [
#         ["Senior", "Java", "no", "no"],
#         ["Senior", "Java", "no", "yes"],
#         ["Mid", "Python", "no", "no"],
#         ["Junior", "Python", "no", "no"],
#         ["Junior", "R", "yes", "no"],
#         ["Junior", "R", "yes", "yes"],
#         ["Mid", "R", "yes", "yes"],
#         ["Senior", "Python", "no", "no"],
#         ["Senior", "R", "yes", "no"],
#         ["Junior", "Python", "yes", "no"],
#         ["Senior", "Python", "yes", "yes"],
#         ["Mid", "Python", "no", "yes"],
#         ["Mid", "Java", "yes", "no"],
#         ["Junior", "Python", "no", "yes"]
#     ]

#     tree_interview = \
#         ["Attribute", "att0",
#             ["Value", "Junior", 
#                 ["Attribute", "att3",
#                     ["Value", "no", 
#                         ["Leaf", "True", 3, 5]
#                     ],
#                     ["Value", "yes", 
#                         ["Leaf", "False", 2, 5]
#                     ]
#                 ]
#             ],
#             ["Value", "Mid",
#                 ["Leaf", "True", 4, 14]
#             ],
#             ["Value", "Senior",
#                 ["Attribute", "att2",
#                     ["Value", "no",
#                         ["Leaf", "False", 3, 5]
#                     ],
#                     ["Value", "yes",
#                         ["Leaf", "True", 2, 5]
#                     ]
#                 ]
#             ]
#         ]

#     y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
#     treeClassifier.fit(X_train_interview, y_train_interview)

#     assert treeClassifier.tree == tree_interview # TODO: fix this


#     X_train_iphone = [
#         ["1", "3", "fair"],
#         ["1", "3", "excellent"],
#         ["2", "3", "fair"],
#         ["2", "2", "fair"],
#         ["2", "1", "fair"],
#         ["2", "1", "excellent"],
#         ["2", "1", "excellent"],
#         ["1", "2", "fair"],
#         ["1", "1", "fair"],
#         ["2", "2", "fair"],
#         ["1", "2", "excellent"],
#         ["2", "2", "excellent"],
#         ["2", "3", "fair"],
#         ["2", "2", "excellent"],
#         ["2", "3", "fair"]
#     ]

#     tree_iphone = \
#         ["Attribute", "att0",
#             ["Value", "1", 
#                 ["Attribute", "att1",
#                     ["Value", "1", 
#                         ["Leaf", "yes", 1, 5]
#                     ],
#                     ["Value", "2",
#                         ["Attribute", "att2", 
#                             ["Value", "excellent",
#                                 ["Leaf", "yes", 1, 2]
#                             ],
#                             ["Value", "fair", 
#                                 ["Leaf", "no", 1, 2]
#                             ]
#                         ]
#                     ],
#                     ["Value", "3", 
#                         ["Leaf", "no", 2, 5]
#                     ]
#                 ]
#             ],
#             ["Value", "2",
#                 ["Attribute", "att2",
#                     ["Value", "excellent",
#                         ["Leaf", "no", 4, 4]
#                     ],
#                     ["Value", "fair",
#                         ["Leaf", "yes", 6, 10]
#                     ]
#                 ]
#             ]
#         ]

#     y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

#     treeClassifier.fit(X_train_iphone, y_train_iphone)
    
#     assert treeClassifier.tree == tree_iphone

#     header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
#     X_train_degrees = [
#         ['A', 'B', 'A', 'B', 'B'],
#         ['A', 'B', 'B', 'B', 'A'],
#         ['A', 'A', 'A', 'B', 'B'],
#         ['B', 'A', 'A', 'B', 'B'],
#         ['A', 'A', 'B', 'B', 'A'],
#         ['B', 'A', 'A', 'B', 'B'],
#         ['A', 'B', 'B', 'B', 'B'],
#         ['A', 'B', 'B', 'B', 'B'],
#         ['A', 'A', 'A', 'A', 'A'],
#         ['B', 'A', 'A', 'B', 'B'],
#         ['B', 'A', 'A', 'B', 'B'],
#         ['A', 'B', 'B', 'A', 'B'],
#         ['B', 'B', 'B', 'B', 'A'],
#         ['A', 'A', 'B', 'A', 'B'],
#         ['B', 'B', 'B', 'B', 'A'],
#         ['A', 'A', 'B', 'B', 'B'],
#         ['B', 'B', 'B', 'B', 'B'],
#         ['A', 'A', 'B', 'A', 'A'],
#         ['B', 'B', 'B', 'A', 'A'],
#         ['B', 'B', 'A', 'A', 'B'],
#         ['B', 'B', 'B', 'B', 'A'],
#         ['B', 'A', 'B', 'A', 'B'],
#         ['A', 'B', 'B', 'B', 'A'],
#         ['A', 'B', 'A', 'B', 'B'],
#         ['B', 'A', 'B', 'B', 'B'],
#         ['A', 'B', 'B', 'B', 'B']
#     ]
#     y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
#                     'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
#                     'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
#                     'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
#                     'SECOND', 'SECOND']

#     tree_degrees = \
#         ["Attribute", "att0",
#             ["Value", "A", 
#                 ["Attribute", "att4",
#                     ["Value", "A", 
#                         ["Leaf", "FIRST", 5, 14]
#                     ],
#                     ["Value", "B",
#                         ["Attribute", "att3",
#                             ["Value", "A", 
#                                 ["Attribute", "att1", 
#                                     ["Value", "A", 
#                                         ["Leaf", "FIRST", 1, 2]
#                                     ],
#                                     ["Value", "B",
#                                         ["Leaf", "SECOND", 1, 2]
#                                     ]
#                                 ]
#                             ],
#                             ["Value", "B",
#                                 ["Leaf", "SECOND", 7, 9]
#                             ]
#                         ]
#                     ]
#                 ]
#             ],
#             ["Value", "B", 
#                 ["Leaf", "SECOND", 12,26]
#             ]
#         ]

#     treeClassifier.fit(X_train_degrees, y_train_degrees)

#     assert treeClassifier.tree == tree_degrees

# def test_decision_tree_classifier_predict():
#     treeClassifier = MyDecisionTreeClassifier()

#     X_train_interview = [
#         ["Senior", "Java", "no", "no"],
#         ["Senior", "Java", "no", "yes"],
#         ["Mid", "Python", "no", "no"],
#         ["Junior", "Python", "no", "no"],
#         ["Junior", "R", "yes", "no"],
#         ["Junior", "R", "yes", "yes"],
#         ["Mid", "R", "yes", "yes"],
#         ["Senior", "Python", "no", "no"],
#         ["Senior", "R", "yes", "no"],
#         ["Junior", "Python", "yes", "no"],
#         ["Senior", "Python", "yes", "yes"],
#         ["Mid", "Python", "no", "yes"],
#         ["Mid", "Java", "yes", "no"],
#         ["Junior", "Python", "no", "yes"]
#     ]

#     y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
#     treeClassifier.fit(X_train_interview, y_train_interview)
#     X_test = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]

#     assert treeClassifier.predict(X_test) == ["True", "False"]

#     X_train_iphone = [
#         ["1", "3", "fair"],
#         ["1", "3", "excellent"],
#         ["2", "3", "fair"],
#         ["2", "2", "fair"],
#         ["2", "1", "fair"],
#         ["2", "1", "excellent"],
#         ["2", "1", "excellent"],
#         ["1", "2", "fair"],
#         ["1", "1", "fair"],
#         ["2", "2", "fair"],
#         ["1", "2", "excellent"],
#         ["2", "2", "excellent"],
#         ["2", "3", "fair"],
#         ["2", "2", "excellent"],
#         ["2", "3", "fair"]
#     ]

#     y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

#     treeClassifier.fit(X_train_iphone, y_train_iphone)
#     X_test = [["2", "2", "fair"], ["1", "1", "excellent"]]
    
#     assert treeClassifier.predict(X_test) == ["yes", "yes"]


#     X_train_degrees = [
#         ['A', 'B', 'A', 'B', 'B'],
#         ['A', 'B', 'B', 'B', 'A'],
#         ['A', 'A', 'A', 'B', 'B'],
#         ['B', 'A', 'A', 'B', 'B'],
#         ['A', 'A', 'B', 'B', 'A'],
#         ['B', 'A', 'A', 'B', 'B'],
#         ['A', 'B', 'B', 'B', 'B'],
#         ['A', 'B', 'B', 'B', 'B'],
#         ['A', 'A', 'A', 'A', 'A'],
#         ['B', 'A', 'A', 'B', 'B'],
#         ['B', 'A', 'A', 'B', 'B'],
#         ['A', 'B', 'B', 'A', 'B'],
#         ['B', 'B', 'B', 'B', 'A'],
#         ['A', 'A', 'B', 'A', 'B'],
#         ['B', 'B', 'B', 'B', 'A'],
#         ['A', 'A', 'B', 'B', 'B'],
#         ['B', 'B', 'B', 'B', 'B'],
#         ['A', 'A', 'B', 'A', 'A'],
#         ['B', 'B', 'B', 'A', 'A'],
#         ['B', 'B', 'A', 'A', 'B'],
#         ['B', 'B', 'B', 'B', 'A'],
#         ['B', 'A', 'B', 'A', 'B'],
#         ['A', 'B', 'B', 'B', 'A'],
#         ['A', 'B', 'A', 'B', 'B'],
#         ['B', 'A', 'B', 'B', 'B'],
#         ['A', 'B', 'B', 'B', 'B']
#     ]
#     y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
#                     'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
#                     'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
#                     'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
#                     'SECOND', 'SECOND']


#     treeClassifier.fit(X_train_degrees, y_train_degrees)
#     X_test = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]

#     assert treeClassifier.predict(X_test) == ["SECOND", "FIRST", "FIRST"]
