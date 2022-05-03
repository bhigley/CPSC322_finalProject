##############################################
# Programmer: Matthew Moore
# Class: CptS 322-01, Spring 2022
# Programming Assignment # 7
# 2/9/2022
# 
# 
# Description: This program runs tests against the myclassifiers API. 
# All of these classifiers are built to be trained in order to make predictions on a dataset.
##############################################

import numpy as np
import mysklearn.myutils as myutils
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.linear_model import LinearRegression
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MyRandomForestClassifier, MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier,\
    MyNaiveBayesClassifier,\
    MyDecisionTreeClassifier


# interview dataset
interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
interview_tree = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

bramer_solution_tree = ['Attribute', 'att0', 
                            ['Value', 'A', 
                                ['Attribute', 'att4', 
                                    ['Value', 'A', 
                                        ['Leaf', 'FIRST', 5, 14]
                                    ], 
                                    ['Value', 'B', 
                                        ['Attribute', 'att3', 
                                            ['Value', 'A', 
                                                ['Attribute', 'att1', 
                                                    ['Value', 'A', 
                                                        ['Leaf', 'FIRST', 1, 2]
                                                    ], 
                                                    ['Value', 'B', 
                                                        ['Leaf', 'SECOND', 1, 2]
                                                    ]
                                                ]
                                            ],
                                            ['Value', 'B', 
                                                ['Leaf', 'SECOND', 7, 9]
                                            ]
                                        ]
                                    ]
                                ]
                            ], 
                            ['Value', 'B', 
                                ['Leaf', 'SECOND', 12, 26]
                            ]
                        ]

iphone_solution_tree = ['Attribute', 'att0', 
                            ['Value', 1, 
                                ['Attribute', 'att1', 
                                    ['Value', 1, 
                                        ['Leaf', 'yes', 1, 5]
                                    ], 
                                    ['Value', 2, 
                                        ['Attribute', 'att2', 
                                            ['Value', 'excellent', 
                                                ['Leaf', 'yes', 1, 2]
                                            ],
                                            ['Value', 'fair', 
                                                ['Leaf', 'no', 1, 2]
                                            ]
                                        ]
                                    ],
                                    ['Value', 3, 
                                        ['Leaf', 'no', 2, 5]
                                    ]
                                ]
                            ],
                            ['Value', 2, 
                                ['Attribute', 'att2', 
                                    ['Value', 'excellent', 
                                        ['Leaf', 'no', 4, 10]
                                    ],
                                    ['Value', 'fair', 
                                        ['Leaf', 'yes', 6, 10]
                                    ]
                                ]
                            ]
                        ]

# bramer degrees dataset
degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

degrees_tree = [] # TODO: figure out what this is by finishing Bramer 5.5 Self-assessment exercise 1

# in-class Naive Bayes example (lab task #1)
inclass_example_col_names = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# RQ5 (fake) iPhone purchases dataset
iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
y_train_iphone = ["no","no","yes","yes","yes","no","yes","no","yes","yes","yes","yes","yes","no","yes"]

# Bramer 3.2 train dataset
time_col_names = ["day", "season", "wind", "rain", "class"]
time_table = [
    ["weekday", "spring", "none", "none", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "none", "slight", "on time"],
    ["weekday", "winter", "high", "heavy", "late"],
    ["saturday", "summer", "normal", "none", "on time"],
    ["weekday", "autumn", "normal", "none", "very late"],
    ["holiday", "summer", "high", "slight", "on time"],
    ["sunday", "summer", "normal", "none", "on time"],
    ["weekday", "winter", "high", "heavy", "very late"],
    ["weekday", "summer", "none", "slight", "on time"],
    ["saturday", "spring", "high", "heavy", "cancelled"],
    ["weekday", "summer", "high", "slight", "on time"],
    ["saturday", "winter", "normal", "none", "late"],
    ["weekday", "summer", "high", "none", "on time"],
    ["weekday", "winter", "normal", "heavy", "very late"],
    ["saturday", "autumn", "high", "slight", "on time"],
    ["weekday", "autumn", "none", "heavy", "on time"],
    ["holiday", "spring", "normal", "slight", "on time"],
    ["weekday", "spring", "normal", "none", "on time"],
    ["weekday", "spring", "normal", "slight", "on time"]
] # 20 newsgroups dataset

# 4 instance table
header4 = ["Acid Durability","Strength","Classification"]
X_train4 = [[1,1],[1,0],[.33,0],[0,0]]
y_train4 = ['bad','bad','good','good']
X_test4 = [[.33,1]]
# 8 instance table
header8 = ["att1","att2"]
X_train8 = [[3, 2],[6, 6],[4, 1],[4, 4],[1, 2],[2, 0],[0, 3],[1, 6]]
y_train8 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
X_test8 = [[2,3]]
# Table from bramer text book
header_bramer_example = ["Attribute 1", "Attribute 2"]
X_train_bramer_example = [[0.8, 6.3],[1.4, 8.1],[2.1, 7.4],[2.6, 14.3],[6.8, 12.6],[8.8, 9.8],
    [9.2, 11.6],[10.8, 9.6],[11.8, 9.9],[12.4, 6.5],[12.8, 1.1],[14.0, 19.9],[14.2, 18.5],[15.6, 17.4],
    [15.8, 12.2],[16.6, 6.7],[17.4, 4.5],[18.2, 6.9],[19.0, 3.4],[19.6, 11.1]]
y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
X_test_bramer = [[9.1,11.0]]

def discretize_high_low(y_column):
    """
    This function creates a parallel categorical list to the continuous list passed in, as high or low
    Attributes:
    y_column: (list of numbers) list of numbers to be discretized as high or low. 
    Returns:
    discretized_list: (list of obj) list of strings either 'high' or 'low' based on the value of condition
    """
    SIMPLE_DISC_VAL = 100
    discretized_list = ['high' if numbery >= SIMPLE_DISC_VAL else 'low' for numbery in y_column]
    return discretized_list

def discretize_high_mid_low(y_column):
    """
    This function creates a parallel categorical list to the continuous list passed in, as high or low or mid
    Attributes:
    y_column: (list of numbers) list of numbers to be discretized as high or low or mid
    Returns:
    discretized_list: (list of obj) list of strings either 'high' or 'low' or 'mid' based on the value of condition
    """
    discretized_list = []
    for numbery in y_column:
        if numbery <= 0:
            discretized_list.append('low')
        elif numbery > 0 and numbery <= 50:
            discretized_list.append('mid')
        elif numbery > 50:
            discretized_list.append('high')
    return discretized_list

def test_simple_linear_regression_classifier_fit():
    """
    This tests the accuracy of the MySimpleLinearRegressionClassifier in terms of its fit funciton.
    The goal is to see if the function successfuly calculates the right fitting slope based off
    the X_train and y_train values passed in. 

    We are proving this function up against desk calculations
    """
    test_regressor = MySimpleLinearRegressor()
    test_regressor_classifier = MySimpleLinearRegressionClassifier(discretize_high_low,test_regressor)
    np.random.seed(0)
    X_train = [[val] for val in range(100)]
    y_train = [value[0] * 2 + np.random.normal(0, 25) for value in X_train]
    test_regressor_classifier.fit(X_train,y_train)
    # Here is our desk calculation of the correct solution
    slope_solution = 1.924917458430444
    intercept_solution = 5.211786196055144
    assert np.isclose(test_regressor.slope, slope_solution)
    assert np.isclose(test_regressor.intercept, intercept_solution)
    # Now we can test this against an edge case scenario, as well as a comprehensive and complex one
    # We can also throw in our other discretizer function for fun
    test_regressor_classifier = MySimpleLinearRegressionClassifier(discretize_high_mid_low,test_regressor)
    X_train = []
    y_train = []
    intercept_solution = 10.123456789
    slope_solution = -0.123456789
    for i in range(1000):
        y_train.append((i * slope_solution) + intercept_solution)
        X_train.append([i])
        if len(X_train) > 1:
            test_regressor_classifier.fit(X_train,y_train)
            assert np.isclose(test_regressor.slope, slope_solution)
            assert np.isclose(test_regressor.intercept, intercept_solution)

def test_simple_linear_regression_classifier_predict():
    """
    This tests the accuracy of the MySimpleLinearRegressionClassifier in terms of its predict function.
    The goal is to see if the function successfuly predicts a value of y_predicted for its corresponding
    X_test instance and discretizes this value accordingly based on the function called.

    We are proving this function's correctness with desk calculations
    """
    DISC_VAL = 100
    test_regressor = MySimpleLinearRegressor()
    test_regressor_classifier = MySimpleLinearRegressionClassifier(discretize_high_low,test_regressor)
    # First lets, test where high is only possible and also the first low
    test_regressor.slope = 1
    test_regressor.intercept = 0
    # Creating a x_test list from 0 to 100
    test_all_low = [[val] for val in range(DISC_VAL + 1)]
    all_low = ['low' for val in range(DISC_VAL + 1)]
    test_all_lows = test_regressor_classifier.predict(test_all_low)
    for i in range(DISC_VAL):
        assert test_all_lows[i] == all_low[i]
    # This best fit value is finally 100 which is not low
    assert test_all_lows[DISC_VAL] != all_low[DISC_VAL]
    # Now lets, test where low is only possible ending with a high class value
    # Line of best fit: y = -x + 199
    test_regressor.slope = -1
    test_regressor.intercept = DISC_VAL * 2 - 1
    # Creating X_test list from 0 to 100
    test_all_high = [[val] for val in range(DISC_VAL + 1)]
    all_high = ['high' for val in range(DISC_VAL + 1)]
    test_all_highs = test_regressor_classifier.predict(test_all_high)
    for i in range(DISC_VAL):
        assert test_all_highs[i] == all_high[i]
    # This best fit value is 99 which is not high
    assert test_all_highs[DISC_VAL] != all_high[DISC_VAL]
    # We can now test our more complex discretizer funciton with a similar framework
    DISC_VAL = 51
    test_regressor = MySimpleLinearRegressor()
    test_regressor_classifier = MySimpleLinearRegressionClassifier(discretize_high_mid_low,test_regressor)
    test_regressor.slope = 1
    test_regressor.intercept = 0
    # X_test list from -50 to 51
    test_disc = [[val] for val in range(-DISC_VAL,DISC_VAL + 1)]
    # y_predictions should be 'low' to zero, 'mid' to 50, then 'high' above 50
    solution_disc = ['low' if val <= 0 else 'mid' for val in range(-DISC_VAL,DISC_VAL + 1)]
    test_disc = test_regressor_classifier.predict(test_disc)
    for i in range(DISC_VAL * 2):
        assert test_disc[i] == solution_disc[i]
    # The final value of y_predicted is 51 which is discretized as 'high' not 'mid'
    assert test_disc[DISC_VAL * 2] != solution_disc[DISC_VAL * 2]

def test_kneighbors_classifier_kneighbors():
    """
    This tests the accuracy of the MyKNeighborsClassifier in terms of its kneighbors function.
    The goal is to see if the function successfuly calculates the three closest neighbors to the given
    X_test instances. We utilize euclidian distance sorting on distance to find these top distances and
    indexes.

    We are proving this function up against desk calculations.
    """
    test_kneighbors = MyKNeighborsClassifier()
    # Create our training data performing a desk check
    test_kneighbors.fit(X_train4,y_train4)
    knn_solution = test_kneighbors.kneighbors(X_test4)
    actual_solution = [[0.67,1.0,1.05]],[[0,2,3]]
    np.allclose(knn_solution,actual_solution)
    # Now let's test against a slightly larger data set with the known result from sklearn
    test_kneighbors.fit(X_train8,y_train8)
    knn_solution = test_kneighbors.kneighbors(X_test8)
    # Now, we can acquire our proven values to test against, we know the sklearn solution...
    sk_distances = [[[1.4142135623730951,1.41421356237309516,2.0]],[[0,4,6]]]
    np.allclose(knn_solution,sk_distances)
    # We can now test from Bramer which is a more complex problem
    test_kneighbors = MyKNeighborsClassifier(5) # Bramers solution looks at the top 5 closest neighbors
    test_kneighbors.fit(X_train_bramer_example,y_train_bramer_example)
    knn_solution = test_kneighbors.kneighbors(X_test_bramer)
    # Here is the desk solution we found through the bramer text
    sk_distances = [[[0.608,1.237,2.202,2.802,2.915]],[[6,5,7,4,8]]]
    np.allclose(knn_solution,sk_distances)

def test_kneighbors_classifier_predict():
    """
    This tests the accuracy of the MyKNeighborsClassifier in terms of its predict function.
    The goal is to see if the function correctly makes a prediction based on majority y_train
    values that the k closest neighbors are in parallel with.

    We are proving this function up against desk calculations
    """
    test_kneighbors = MyKNeighborsClassifier()
    # Testing against a 4 instance data set first
    test_kneighbors.fit(X_train4,y_train4)
    my_solution = test_kneighbors.predict(X_test4)
    assert my_solution == ['good'] # Desk calculation tells us that the majority of neighbors are classified good
    # Now let's test against a data set of 8 instances
    test_kneighbors.fit(X_train8,y_train8)
    my_solution = test_kneighbors.predict(X_test8)
    assert my_solution == ['yes'] # Desk calculation tells us that the majority of neighbors are classified yes
    # Finally lets test against the more complex Bramer Data,set
    test_kneighbors.fit(X_train_bramer_example,y_train_bramer_example)
    my_solution = test_kneighbors.predict(X_test_bramer)
    assert my_solution == ['+'] # Neighbors classified with 3 '+' and 2'-', so we know the majority solution

def test_dummy_classifier_fit():
    """
    Here we are simply testing the MyDummyClassifier class to see if the fit function successfully 
    stores the value of the most_common_label as the correct most common class label. This value should
    always be the majority y_train attributes of the columns

    Our tested function will be put up against the hard coded solution which is very clearly evident
    """
    test_dummy = MyDummyClassifier()
    # First we can test with no X_train and a high rate of a certain class label in y_train
    X_train = []
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    test_dummy.fit(X_train,y_train)
    # "Yes" is the dominant class label so, the stored value in most_common_label must be "yes"
    assert test_dummy.most_common_label == "yes"
    # Now we can test with some X_train and a greater dispersion of class labels in y_train
    X_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    y_train = y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    test_dummy.fit(X_train,y_train)
    assert test_dummy.most_common_label == "no"
    y_train = y_train = list(np.random.choice(["a","b","c","d","e","f","g","h","i","j"],
                         1000, replace=True, p=[0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.19]))
    test_dummy.fit(X_train,y_train)
    assert test_dummy.most_common_label == "j"

def test_dummy_classifier_predict():
    """
    Here we are simply testing the MyDummyClassifier class to see if the predict function successfully 
    predicts simply whatever the stored most_common_label has stored for as many X_test instances
    there are in the data set. 

    Our tested function will be put up against the hard coded solution which is very clearly evident
    """
    test_dummy = MyDummyClassifier()
    # First we can test with no X_train and a high rate of a certain class label in y_train
    X_train = []
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    test_dummy.fit(X_train,y_train)
    # "Yes" is the dominant class label so, this is the prediction no matter what
    assert test_dummy.predict([[1.0]]) == ["yes"]
    # Now we can test with some predict with more test instances
    X_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    y_train = y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    test_dummy.fit(X_train,y_train)
    assert test_dummy.predict([[1.0],[0],[100],[1000],[20],[15]])  == ["no","no","no","no","no","no"]
    y_train = y_train = list(np.random.choice(["a","b","c","d","e","f","g","h","i","j"],
                         1000, replace=True, p=[0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.09,0.19]))
    test_dummy.fit(X_train,y_train)
    assert test_dummy.predict([[1.0]])  == ['j']


def test_naive_bayes_classifier_fit():
    # TEST 1: 8 instance training set example traced in class on the iPad
    # asserting against our desk check of the priors and posteriors
    # all desk checks have a clear alphabetical order providing easy comparison
    correct_priors = [3/8,5/8]
    correct_posteriors = [["class",1,2,5,6],["no",2/3,1/3,2/3,1/3],["yes",4/5,1/5,2/5,3/5]]
    mynbc = MyNaiveBayesClassifier()
    mynbc.fit(X_train_inclass_example,y_train_inclass_example)
    assert mynbc.priors == correct_priors
    assert mynbc.posteriors == correct_posteriors
    # TEST 2: Use the 15 instance training set example from RQ5
    # asserting against your desk check of the priors and posteriors
    correct_priors = [5/15,10/15]
    correct_posteriors = [["class",1,2,1,2,3,"excellent","fair"],
                          ["no",3/5,2/5,1/5,2/5,2/5,3/5,2/5],
                          ["yes",2/10,8/10,3/10,4/10,3/10,3/10,7/10]]
    mynbc.fit(X_train_iphone,y_train_iphone)
    assert mynbc.priors == correct_priors
    assert mynbc.posteriors == correct_posteriors
    # TEST 3: Use Bramer 3.2 Figure 3.1 train dataset example
    # asserting against the priors and posteriors solution in Figure 3.2.
    correct_priors = [1/20,2/20,14/20,3/20]
    correct_posteriors = [["class","holiday","saturday","sunday","weekday","autumn","spring","summer",
                        "winter","high","none","normal","heavy","none","slight"],
                       ["cancelled",0/1,1/1,0/1,0/1,0/1,1/1,0/1,0/1,1/1,0/1,0/1,1/1,0/1,0/1],
                            ["late",0/2,1/2,0/2,1/2,0/2,0/2,0/2,2/2,1/2,0/2,1/2,1/2,1/2,0/2],
                         ["on time",2/14,2/14,1/14,9/14,2/14,4/14,6/14,2/14,4/14,5/14,5/14,1/14,5/14,8/14],
                       ["very late",0/3,0/3,0/3,3/3,1/3,0/3,0/3,2/3,1/3,0/3,2/3,2/3,1/3,0/3]]
    mynbc.fit(myutils.table_by_columns(time_table,time_col_names,time_col_names[:-1]),
                myutils.get_column(time_table,time_col_names,time_col_names[-1]))
    assert mynbc.priors == correct_priors
    assert mynbc.posteriors == correct_posteriors

def test_naive_bayes_classifier_predict():
    # TEST 1: Use the 8 instance training set example traced in class on the iPad
    # asserting against our desk check prediction
    X_test_inclass_example = [[1,5]]
    mynbc = MyNaiveBayesClassifier()
    mynbc.fit(X_train_inclass_example,y_train_inclass_example)
    assert mynbc.predict(X_test_inclass_example) == ["yes"]
    # TEST 2: Use the 15 instance training set example from RQ5
    # asserting against your desk check predictions for the two test instances
    X_test_iphone = [[2,2,"fair"],[1,1,"excellent"]]
    mynbc.fit(X_train_iphone,y_train_iphone)
    assert mynbc.predict(X_test_iphone) == ["yes","no"]
    # TEST 3: Use Bramer 3.2 unseen instance ["weekday", "winter", "high", "heavy"],
    # and Bramer 3.6 Self-assessment exercise 1 unseen instances
    # asserting against the solution prediction on pg. 28-29
    # and the exercise solution predictions in Bramer Appendix E
    X_test_time = [["weekday","winter","high","heavy"]]
    mynbc.fit(myutils.table_by_columns(time_table,time_col_names,time_col_names[:-1]),
                myutils.get_column(time_table,time_col_names,time_col_names[-1]))
    assert mynbc.predict(X_test_time) == ["very late"]

def test_decision_tree_classifier_fit():
    # TEST 1: Use the 14 instance "interview" training set example traced in class on the iPad
    mydecision_tree = MyDecisionTreeClassifier()
    mydecision_tree.fit(myutils.table_by_columns(interview_table,interview_header,interview_header[:-1]),
                            myutils.get_column(interview_table,interview_header,interview_header[-1]))
    assert mydecision_tree.tree == interview_tree
    # TEST 2: Use the Bramer 4.1 Figure 4.3 degrees dataset example, 
    # asserting against the tree you create when you finish Bramer 5.5 Self-assessment exercise 1
    mydecision_tree.fit(myutils.table_by_columns(degrees_table,degrees_header,degrees_header[:-1]),
                            myutils.get_column(degrees_table,degrees_header,degrees_header[-1]))
    assert mydecision_tree.tree == bramer_solution_tree
    # TEST3: Use the 15 instance "iPhone" training set example from RQ5
    mydecision_tree.fit(X_train_iphone,y_train_iphone)
    assert mydecision_tree.tree == iphone_solution_tree
def test_decision_tree_classifier_predict():
    X_test_interview = [["Junior", "Java", "yes", "no"],["Junior", "Java", "yes", "yes"]]
    X_test_bramer = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
    X_test_iphone = [[2,2,"fair"],[1,1,"excellent"]]
    # TEST 1: Use the 14 instance "interview" training set example traced in class on the iPad
    mydecision_tree = MyDecisionTreeClassifier()
    mydecision_tree.fit(myutils.table_by_columns(interview_table,interview_header,interview_header[:-1]),
                            myutils.get_column(interview_table,interview_header,interview_header[-1]))
    assert mydecision_tree.predict(X_test_interview) == ["True","False"]
    # TEST 2: Use the Bramer 4.1 Figure 4.3 degrees dataset example, 
    # asserting against the tree you create when you finish Bramer 5.5 Self-assessment exercise 1
    mydecision_tree.fit(myutils.table_by_columns(degrees_table,degrees_header,degrees_header[:-1]),
                            myutils.get_column(degrees_table,degrees_header,degrees_header[-1]))
    assert mydecision_tree.predict(X_test_bramer) == ["SECOND","FIRST","FIRST"]
    # TEST3: Use the 15 instance "iPhone" training set example from RQ5
    mydecision_tree.fit(X_train_iphone,y_train_iphone)
    assert mydecision_tree.predict(X_test_iphone) == ["yes","yes"]