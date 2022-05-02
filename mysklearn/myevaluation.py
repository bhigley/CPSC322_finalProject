#############################################
# Programmer: Matthew Moore
# Class: CptS 322-01, Spring 2022
# Programming Assignment # 6
# 2/9/2022
# 
# Description: This program contains functions for various methods of splitting up data into
# training and testing. Some of them return list of lists for multiple iterations of testing to 
# have a more comprehensive set of testing data while the bootsrap method does not really do this.
# The final two functions are built for analysis and visualization purposes so we can interpret
# what our metric are like. 
#
# We can utilize this functionality for our mypytable to better split up our data allowing us to run better
# predictions
##############################################

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    if test_size < 1: # Case of percentage
        cutoff_index = (len(X) * (1 - test_size)) - 1
    else: # Case of numeric test cases
        cutoff_index = len(X) - test_size

    if shuffle:
        myutils.randomize_in_place(X,y,random_state)
    for instance_index in range(len(X)):
        if instance_index >= cutoff_index: # Remaining instances for testing
            X_test.append(X[instance_index])
            y_test.append(y[instance_index])
        else: # First set of instances for training
            X_train.append(X[instance_index])
            y_train.append(y[instance_index])

    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_indexes = list(range(len(X)))
    if shuffle:
        myutils.randomize_in_place(X,X_indexes,random_state=random_state) # Track original indexes
    first_num_folds = len(X) % n_splits
    first_fold_size = len(X) // n_splits + 1
    other_fold_size = len(X) // n_splits
    other_num_folds = n_splits - first_num_folds
    curr_index = 0
    thefold = []
    thefolds = []
    for fold in range(first_num_folds): # Create first larger folds
        for instance in range(first_fold_size):
            thefold.append(X_indexes[curr_index])
            curr_index += 1
        thefolds.append(thefold)
        thefold = []
    for fold in range(other_num_folds): # Create remaining smaller folds if necessary
        for instance in range(other_fold_size):
            thefold.append(X_indexes[curr_index])
            curr_index += 1
        thefolds.append(thefold)
        thefold = []
    X_train_folds, X_test_folds = myutils.split_folds_to_train_test(thefolds)
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X_indexes = list(range(len(X)))
    if shuffle:
        myutils.randomize_in_place(X_indexes,y,random_state=random_state) # Track original indexes
    y_indexed = []
    for i, y_instance in enumerate(y):
        y_indexed.append([X_indexes[i],y_instance])
    # Create subtables containing original indexes and y-classifications
    group_names, group_subtables = myutils.group_by(y_indexed,["x indexes","instance"],"instance") 
    curr_index = 0
    folds = [[] for split in range(n_splits)]
    # Set up the folds and one by one place indexes into each fold
    for group_subtable in group_subtables:
        for instance in group_subtable:
            folds[curr_index % n_splits].append(instance[0]) # % brings us back to fold zero with each round 
            curr_index += 1
    X_train_folds, X_test_folds = myutils.split_folds_to_train_test(folds)
    return X_train_folds, X_test_folds
    

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    X_sample, X_out_of_bag, X_indexes_used = list(), list(), list()
    if y != None:
        y_sample, y_out_of_bag = list(), list()
    else:
        y_sample, y_out_of_bag = None, None
    n_samples = len(X) if n_samples == None else n_samples
    random_indexes = myutils.randints(len(X),n_samples,random_state)
    for sample in range(n_samples):
        X_sample.append(X[random_indexes[sample]])
        if y != None:
            y_sample.append(y[random_indexes[sample]])
        X_indexes_used.append(random_indexes[sample])
    for i, X_instance in enumerate(X):
        if i not in X_indexes_used:
            X_out_of_bag.append(X[i])
            if y != None:
                y_out_of_bag.append(y[i])
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix_row,matrix = list(), list()
    for label_actual in labels:
        for label_predicted in labels:
            num_cases = 0
            for index in range(len(y_true)):
                if y_true[index] == label_actual and y_pred[index] == label_predicted:
                    num_cases += 1
            matrix_row.append(num_cases)
        matrix.append(matrix_row)
        matrix_row = []
    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    num_correct = 0
    for index in range(len(y_true)):
        try:
            if int(y_true[index]) == int(y_pred[index]):
                num_correct += 1
        except:
            if y_true[index] == y_pred[index]:
                num_correct += 1
    if normalize == True:
        return num_correct / len(y_true)
    else:
        return num_correct


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels == None:
        labels = myutils.unique_values(y_true)
    if pos_label == None:
        pos_label = labels[0]
    num_true_positives, num_false_positives = 0, 0
    for index in range(len(y_true)):
        if y_true[index] == y_pred[index] and y_pred[index] == pos_label:
            # Case of a positive label predicted as positive
            num_true_positives += 1
        elif y_true[index] != y_pred[index] and y_pred[index] == pos_label:
            # Case of a negative label predicted as positive
            num_false_positives += 1
    if num_true_positives == 0:
        return 0
    return num_true_positives / (num_true_positives + num_false_positives)

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels == None:
        labels = myutils.unique_values(y_true)
    if pos_label == None:
        pos_label = labels[0]
    num_true_positives, num_false_negatives = 0, 0
    for index in range(len(y_true)):
        if y_true[index] == y_pred[index] and y_pred[index] == pos_label:
            # Case of a positive label predicted as positive
            num_true_positives += 1
        elif y_true[index] != y_pred[index] and y_pred[index] != pos_label:
            # Case of a positive label predicted as a negative, false negative
            num_false_negatives += 1
    if num_true_positives == 0:
        return 0
    return num_true_positives / (num_true_positives + num_false_negatives)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    if labels == None:
        labels = myutils.unique_values(y_true)
    if pos_label == None:
        pos_label = labels[0]
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
