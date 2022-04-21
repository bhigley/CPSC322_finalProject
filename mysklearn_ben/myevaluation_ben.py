from re import X
import myutils_ben as myutils
import myutils_knn as myutils_knn
import numpy as np

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
    np.random.seed(random_state)
    X_test = []
    y_test = []
    X_train = []
    y_train = []
    # calulate number of items to be taken out of training set
    if shuffle is True:
        myutils.randomize_in_place(X, y)
    if test_size < 1: # converts test_size to a float if it is a percentage
        test_size = len(X) * test_size
    i = 0
    while i <= len(X) - test_size - 1:
        X_train.append(X[i])
        y_train.append(y[i])
        i += 1
    while i < len(X):
        X_test.append(X[i])
        y_test.append(y[i])
        i += 1

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
    np.random.seed(random_state)
    X_train = []
    X_test_folds = []
    X_train_folds = []
    indexes = [] # stores index values that will be added to X_train
    folds = [] # stores a temporary fold that will be added to a list
    for i in range(len(X)): # makes a parallel list that stores indexes for each item in X
        indexes.append(i)
    if shuffle is True:
        myutils.randomize_in_place(indexes)
    # create folds
    num_first_folds = len(X) % n_splits # first n_samples % n_splits
    size_first_folds = len(X) // n_splits + 1
    i = 0
    fold = []
    # builds first folds
    while len(folds) < num_first_folds:
        while len(fold) < size_first_folds:
            # fold.append(i)
            fold.append(indexes[i])
            i += 1
        folds.append(fold)
        fold = []
    # builds other folds
    while len(folds) < n_splits:
        while len(fold) < size_first_folds - 1: # i.e. n_samples // n_splits
            fold.append(indexes[i])
            i += 1
        folds.append(fold)
        fold = []
    for fold in folds:
        X_test_folds.append(fold)
        X_train = indexes.copy()
        for item in fold:
            if item in X_train:
                X_train.remove(item)
        X_train_folds.append(X_train)
        X_train = []

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
    np.random.seed(random_state)
    X_train = []
    X_test_folds = []
    X_train_folds = []
    folds = []
    indexes = [i for i in range(len(X))]
    if shuffle is True:
        myutils_knn.randomize_in_place(indexes, y) # not sure if i need indexes and y, both pass the test
    deck_one, deck_two = myutils_knn.group_by(indexes, y) # splitting indexes into two
    indexes = deck_one.copy() + deck_two.copy() # the decks will be empty after dealing so need to keep track
    # card decks based on value
    # build folds
    for i in range(n_splits): # append empty fold n_splits times
        folds.append([])
    for num in range(len(X)): # not efficient but whatever
        for fold in folds:
            if len(deck_one) > 0: # deck is not empty
                fold.append(deck_one.pop(0))
            elif len(deck_two) > 0: # deck one is empty
                fold.append(deck_two.pop(0))
            else: # all decks have been dealt
                break
    for fold in folds: # same process as normal kfold
        X_test_folds.append(fold)
        X_train = indexes.copy()
        for item in fold:
            if item in X_train:
                X_train.remove(item)
        X_train_folds.append(X_train)
        X_train = []

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
    np.random.seed(random_state)
    if n_samples is None: # not sure if this is right
        n_samples = len(X)
    indexes = [i for i in range(len(X))] # build index list
    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []
    # randomly select n_samples indexes from indexes
    # selected goes into trainining set
    # check trainng set and if an index is not in it, add it to test set
    for i in range(n_samples):
        index = indexes[np.random.randint(0, len(X))]
        X_sample.append(X[index])
        if y is not None:
            y_sample.append(y[index])
    for item in indexes:
        if X[item] not in X_sample:
            X_out_of_bag.append(X[item])
            if y is not None:
                y_out_of_bag.append(y[item])
    if y is None:
        y_sample = None
        y_out_of_bag = None


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
    matrix = np.zeros((len(labels), len(labels)))
    i = 0
    while i < len(labels): # while there is another row
        for j in range(len(labels)):
            for index in range(len(y_true)):
                if y_true[index] == labels[i] and y_pred[index] == labels[j]:
                    matrix[i][j] += 1
        i += 1

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
    correct_pred = 0
    incorrect_pred = 0
    accuracy = 0
    for i in range(len(y_true)):
        if y_true[i] is y_pred[i]:
            correct_pred += 1
        else:
            incorrect_pred += 1
    if normalize is True:
        accuracy = correct_pred / (correct_pred + incorrect_pred)
    else:
        accuracy = correct_pred

    return accuracy
