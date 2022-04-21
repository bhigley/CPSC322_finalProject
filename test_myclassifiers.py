##############################################
# Programmer: Matthew Moore
# Class: CptS 322-01, Spring 2022
# Programming Assignment #4
# 2/9/2022
# 
# 
# Description: This program runs tests against the myclassifiers API. 
# All of these classifiers are built to be trained in order to make predictions on a dataset.
##############################################

import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier
from sklearn.linear_model import LinearRegression

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
