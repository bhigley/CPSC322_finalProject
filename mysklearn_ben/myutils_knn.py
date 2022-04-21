from distutils.log import error
import numpy as np

def randomize_in_place(alist, parallel_list=None):
    """Reorganizes the list without losing index or parallel list value

        Args:
            alist (list) : holds the data to be reorganized
            parallel_list : holds corresponding class

        Returns:
            none

        Notes:
            there doesn't have to be a parallel list
        """
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]

def group_by(indexes, y):
    """Groups the values into two lists of indexes

        Args:
            indexes (int) : holds the indexes for the values
            y : stores the data that is used to split into groups

        Returns:
            group_one_indexes, group_two_indexes

        Notes:
            like splitting cards into two decks based on color
        """
    groups = []
    group_one_y = [] # stores first y values
    group_two_y = [] # stores second y values
    group_one_indexes = [] # stores first x values
    group_two_indexes = [] # stores second x values
    for item in y:
        if item not in groups:
            groups.append(item)
    i = 0
    for item in y:
        if item == groups[0]:
            group_one_y.append(item)
            group_one_indexes.append(indexes[i])
        else:
            group_two_y.append(item)
            group_two_indexes.append(indexes[i])
        i += 1
    return group_one_indexes, group_two_indexes

def get_random_indexes(num_rands, lower, upper):
    """Gets the number of instances of a value from a table

        Args:
            num_rands (int): stores the number of random numbers
                wanted
            lower (int): the lower bound for randint
            upper (int): upper bound for randint

        Returns:
            rand_list :list holding the requested number of random ints
        """
    np.random.seed(0)
    rand_list = []
    for i in range(num_rands):
        rand_list.append(np.random.randint(lower, upper))
    return rand_list

def discretize_high_low(y_values):
    """Discretizes into high and low bins

        Args:
            y_values (list of floats): is converted on swivle point of 100

        Returns:
            labels (list of strings): list made up of corresponding
                high or low label
        """
    labels = []
    for val in y_values:
        if val >= 100:
            labels.append("high")
        else:
            labels.append("low")
    return labels

def compute_euclidean_distance(v1, v2):
    """Uses euclidian formula to calculate distance

        Args:
            v1 (list)
            v2 (list)

        Returns:
            distance

        Notes:
            v1 and v2 are parallel lists
        """
    print(v1, v2)
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v2))]))

def normalize_values(X_train, X_test):
    """Normalizes the values passed in

        Args:
            X_train (list of lists): holds the x data for the training set
            X_test (list of lists): holds the x data for the testing set

        Returns:
            X_train, X_test
        Notes:
            returned with updated values following normalization rules
        """
    for i in range(len(X_train[0])): # iterates trhough each attribute
        min_val = X_train[0][i]
        max_val = X_train[0][i]
        for row in X_train:
            if row[i] > max_val:
                max_val = row[i]
            if row[i] < min_val:
                min_val = row[i]
        ran = max_val - min_val
        if ran != 0.0: # new
            for row in X_train:
                row[i] = (row[i] - min_val) / ran
            for item in X_test:
                item[i] = (item[i] - min_val) / ran
        ran = 0

    return X_train, X_test

def get_column(table, col_index):
    """Extracts a column from a mypytable object

        Args:
            table (Mypytable()): holds the data from a csv file

        Returns:
            col (list): contains the values extraceted from the table at column

        Notes:
            Changed to NA for vgsales.csv file
        """
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def get_frequencies(data):
    """Gets the number of instances of a value from a table

        Args:
            table (2D list): holds the data from a csv file
            header (list): contains the column names in table
            col_name (string): the name of the column to be counted

        Returns:
            two lists containg the item and its frequency
        """
    col = data
    col.sort() # inplace
    # # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)

    return values, counts # we can return multiple values in python

def discretize_auto(y_values):
    """Discretizes into bins based on mpg

        Args:
            y_values (list of floats):

        Returns:
            labels (list of strings): list made up of corresponding
                mpg label range(1 - 10)
        """
    labels = []
    for val in y_values:
        if val <= 13:
            labels.append("1")
        elif val <= 14:
            labels.append("2")
        elif val <= 16:
            labels.append("3")
        elif val <= 19:
            labels.append("4")
        elif val <= 23:
            labels.append("5")
        elif val <= 26:
            labels.append("6")
        elif val <= 30:
            labels.append("7")
        elif val <= 36:
            labels.append("8")
        elif val <= 44:
            labels.append("9")
        else:
            labels.append("10")
    return labels