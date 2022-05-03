#############################################
# Programmer: Matthew Moore
# Class: CptS 322-01, Spring 2022
# Programming Assignment # 6
# 3/30/2022
# 
# Description: This program holds all of the utility functions utilized in the entire project.
# these functions include all numpy and list-based helper functions, as well as instances of needing
# random numbers. Miscellaneous items here. 
##############################################

from platform import node
import numpy as np
import math
import graphviz as gv

def compute_random_subset(values, num_values):
    # used for F in RandomForest
    # there is a function np.random.choice()
    values_copy = values[:] # shallow copy
    np.random.shuffle(values_copy) # in place shuffle
    return values_copy[:num_values]

def majority_vote(att_partition): # working for basic 1 element list
    """Used to determine a clash

        Args:
            att_partition : the attributes partitioned
        Returns:
            the majority vote

        Notes:
            needed for clashes in decision tree
        """
    majority = att_partition[0][-1]
    majority_count = 0
    for vote in att_partition:
        vote_count = 0
        for other_vote in att_partition:
            if vote[-1] == other_vote[-1]:
                vote_count += 1
        if vote_count > majority_count:
            majority = vote[-1]
            majority_count = vote_count
    return majority

def create_table_from_parallel_lists(original_table,list_of_parallel_lists):
    new_cols = []
    new_cols_inner = []
    for index in range(len(original_table.data)):
        for parallel_list in list_of_parallel_lists:
            stats_cols_inner.append(parallel_list[index])
        new_cols.append(stats_cols_inner)
        stats_cols_inner = []
    return new_cols

def print_decision_rules_helper(tree):
    if tree[0] == "Leaf": # Leaf node case
        print("THEN",tree[1])
        return tree
    else:
        if tree[0] == "Attribute":
            print("IF",tree[1],"=",tree[2][1],"AND", end=" ")
        for node_index in range(len(tree)):
            if node_index > 1:
                print_decision_rules_helper(tree[node_index])
                if tree[0] == "Attribute":
                    print("IF",tree[1],"=",tree[2][1],"AND", end=" ")

def convert_header_to_string(table):
    """ When the header is apart of the table, this is a good function to change just the header to a string
    """
    new_header = []
    new_table = []
    for element in table[0]:
        try:
            new_header.append(str(element))
        except:
            new_header.append(element)
    for i, row in enumerate(table):
        if i == 0:
            new_table.append(new_header)
        else:
            new_table.append(row)
    return new_table

def decision_traverse(tree,X_test_instance):
    if tree[0] == "Leaf":
        return tree[1]
    else:
        if tree[0] == "Attribute":
            attribute_index = tree[1][3]
            attribute_value = X_test_instance[int(attribute_index)]
            for node_index in range(len(tree)):
                if node_index > 1:
                    if tree[node_index][0] == "Value":
                        if tree[node_index][1] == attribute_value:
                            return decision_traverse(tree[node_index],X_test_instance)
        return decision_traverse(tree[2],X_test_instance)

def tdidt_predict(header, tree, instance):
    """Used to traverse the decision tree

        Args:
            header : list of strings holding attribute names

        Returns:
            the leaf value for a given instance
        Notes:
            recursive function
        """
    # recursively traverse tree to make a prediction
    # are we at a leaf node (base case) or attribute node?
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1] # label
    # we are at an attribute
    # find attribute value match for instance
    # for loop
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            # we have a match, recurse
            return tdidt_predict(header, value_list[2], instance)
        

def traverse_tree(tree, prev, vis, num):
    if tree[0] == "Leaf": # Leaf node case
        # vis.edge(prev,tree[1],label=tree[1])
        return tree
    else:
        if prev != None:
            if tree[0] == "Value":
                if tree[2][0] == "Attribute":
                    vis.edge(prev,str(tree[2][1]),label=str(tree[1]))
                else: # Case of a leaf node connection
                    vis.edge(prev,str(tree[2][1] + str(num)),label=str(tree[1]))
        else:
            vis.node(tree[1])
        for node_index in range(len(tree)):
            if node_index > 1:
                # TODO: Configure a way to implement numbering leaf nodes down the recursion
                # process and then back up again, test original tree attempt
                traverse_tree(tree[node_index],tree[1],vis,num + 1)
                num += 1

def create_posterior_table(instances,attributes):
    priors, values_list, counts_list, value_list, count_list = [], [], [], [], []
    new_tables = group_by(instances,attributes,attributes[-1])
    header, new_tables = split_on_header(new_tables)
    for new_table in new_tables:
        for attribute_index in range(len(attributes) - 1):
            values, non_counts = get_frequencies(instances,attributes,attributes[attribute_index])
            counts = get_frequencies(new_table,attributes,attributes[attribute_index])
            all_counts = convert_nonpresent_freq_to_zero(values,counts)
            value_list.append(values)
            count_list.append(all_counts)
        priors.append(len(new_table)/len(instances))
        values_list.append(value_list)
        counts_list.append(count_list)
        value_list, count_list = [], []
    return convert_freqs_to_table(values_list,counts_list,header)
    
def select_attribute(instances, attributes):
    original_instances, instances = instances
    # print("INSTANCES",instances)
    entropy_values, entropy_sums = [], []
    used_index = 0
    posterior_table = create_posterior_table(instances, attributes)
    posterior_table = convert_header_to_string(posterior_table)
    values, counts = get_frequencies(instances,attributes,attributes[-1])
    for attribute in attributes:
        if attribute == attributes[-1]:
            continue # Case 1: we don't want to calculate entropy for the class label
        entropy_sum = 0
        attribute_values, attribute_counts = get_frequencies(instances,attributes,attribute)
        for j, attribute_value in enumerate(attribute_values):
            posteriors = get_column(posterior_table,posterior_table[0],attribute_value)
            entropy_value = 0
            for i, count in enumerate(counts):
                # first index of counts will be the total count of this given attribute
                if attribute_counts[j] == 0 or posteriors[i + 1] == 0 or count == 0:
                    entropy_value = 0
                else:
                    entropy_value = entropy_value - ((count * posteriors[i + 1] / attribute_counts[j]) * math.log2(count * posteriors[i + 1] / attribute_counts[j]))
            # print("Individual entropy values:",entropy_value, "posteriors:",posteriors, "count:",counts)
            entropy_sum = entropy_sum + (entropy_value * attribute_counts[j] / len(instances))
            # We must use this syntax below to avoid repeating columns
            posterior_table[0][used_index] = "used"
            used_index += 1
        entropy_sums.append(entropy_sum)
        # print("Attribute:", attribute, "Entropy:", entropy_sum)
    # entropy values and attributes are parallel so this works
    # if there are multiple attributes yielding the same entropy, we will take the first one in the list
    return attributes[entropy_sums.index(min(entropy_sums))], entropy_sums.index(min(entropy_sums))

def all_same_class(instances):
    """This function takes a table and returns True if all of the instances have the same class.
    Accepts, a list of the instances from the table with the final column being the class labels."""
    if len(instances) == 0:
        return True
    initial_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != initial_label: # All instances must be the same as the first one
            return False
    return True

def partition_instances(instances, split_attribute_index):
    original_instances, instances = instances
    # lets use a dictionary
    # TODO: Figure out how to partition on all attributes, even when the current set of attributes does not include
    # all of the possible ones from the original dataset. 
    header = ["attr" + str(i) for i in range(len(instances[0]) - 1)]
    header.append("class")
    partitions = {} # key (string): value (subtable)
    att_domain = unique_values(get_column(original_instances,header,header[split_attribute_index])) # e.g. ["Junior", "Mid", "Senior"]
    for att_value in att_domain:
        partitions[att_value] = []
        for instance in instances:
            if instance[split_attribute_index] == att_value:
                partitions[att_value].append(instance)
    return partitions

def tdidt(current_instances, available_attributes, F=None):
    # basic approach (uses recursion!!):
    # print("available_attributes:", available_attributes)
    # select an attribute to split on
    if F == None:
        attribute, attribute_index = select_attribute(current_instances, available_attributes) # commented out for RandomForest
    else:
        K_subsets = compute_random_subset(available_attributes[:-1], F) # getting random subset without class label
        K_subsets.append(available_attributes[-1]) # adding class label
        # print(available_attributes, "test")
        # print(K_subsets, "ben")
        attribute, attribute_index = select_attribute(current_instances, K_subsets) # added for RandomForest
    # print("splitting on attribute:", attribute, attribute_index)
    original_attributes = available_attributes[:]
    available_attributes.remove(attribute)
    print(original_attributes, available_attributes)
    tree = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, attribute_index)
    original_instances, current_instances = current_instances
    original_instances = remove_column(original_instances,original_instances[0],
                    original_instances[0][attribute_index])
    # print("partitions:", partitions)
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value, att_partition in partitions.items():
        # print("curent attribute value:", att_value, len(att_partition))
        value_subtree = ["Value", att_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            # print("CASE 1 all same class")
            value_subtree.append(["Leaf", att_partition[0][-1],len(att_partition),len(current_instances)])
            tree.append(value_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        # elif len(att_partition) > 0 and len(available_attributes) == 1:
        elif (len(att_partition) > 0 and len(available_attributes) == 0):
            # print("CASE 2 no more attributes")
            class_values, class_counts = get_frequencies(att_partition,original_attributes,original_attributes[-1])
            i = class_counts.index(max(class_counts))
            value_subtree.append(["Leaf",class_values[i],sum(class_counts),len(current_instances)])
            tree.append(value_subtree)
            # subcase of having a mix of labels, handle clash with majority
            # vote leaf node
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            # print("CASE 3 empty partition")
            # "backtrack" to replace the attribute node
            # subtree = tdidt(att_partition, available_attributes.copy())
            # with a majority vote leaf node
            curr_values, curr_counts = get_frequencies(current_instances,original_attributes,original_attributes[-1])
            tree = ["Leaf",
                    curr_values[curr_counts.index(max(curr_counts))], # get the majority vote
                    sum(curr_counts),
                    len(original_instances) - len(current_instances) - 1
                 ]
        else: # the previous conditions are all false... recurse!!
            # print("CASE 4 recurse")
            if len(att_partition[0]) != len(available_attributes):
                att_partition = table_by_columns(att_partition,original_attributes,available_attributes)
            subtree = tdidt([original_instances, att_partition], available_attributes.copy())
            # note the copy
            # append subtree to value_subtree and to tree
            value_subtree.append(subtree)
            tree.append(value_subtree)
            # appropriately
    return tree

def unique_values(value_list):
    """ Returns a list of unique values from the list of values given. These values are sorted
    in ascending order.
    """
    unique_values, ordered_unique_values = [], []
    for value in value_list:
        if value not in unique_values:
            unique_values.append(value)
    return sorted(unique_values)

def convert_nonpresent_freq_to_zero(possible_values,current_counts):
    """ This function takes a list of possible values and a list of current counts for some of these values,
    converting all of the non-present values to zero.
    """
    INDEX_OF_VALUES = 0
    INDEX_OF_COUNTS = 1
    all_counts = []
    current_counts_index = 0
    for i, value in enumerate(possible_values):
        if value in current_counts[INDEX_OF_VALUES]:
            all_counts.append(current_counts[INDEX_OF_COUNTS][current_counts_index])
            current_counts_index += 1
        else:
            all_counts.append(0)
    return all_counts
def convert_freqs_to_table(values_list,counts_list,header):
    """ Converts a list of frequencies into a single table containing posterior values, which is
    the ratio of the counts of a current value to the total counts that have that same class label. 
    These two lists should be parallel. 

    Some indexing is at plus one because we are setting the first item as the header
    """
    new_table = [[] for tables in range(len(counts_list) + 1)]
    new_table[0].append("class") # First column is the classifying attribute always
    for i, attribute_table in enumerate(counts_list):
        new_table[i + 1].append(header[i])
        for j, counts in enumerate(attribute_table):
            for k, count in enumerate(counts):
                if i == 0: # Only setting the header based on values from the first table
                    try: # For integer values we should try to conver them back to ints
                        new_table[0].append(int(values_list[i][j][k]))
                    except:
                        new_table[0].append(values_list[i][j][k])
                new_table[i + 1].append(count / sum(counts))
    return new_table



def split_on_header(table):
    """This function takes a table which includes the header and removes the header from the table
    returning a list for the header and the original table without the header
    """
    new_table = []
    for i, row in enumerate(table):
        if i == 0: # First row
            header = row
        else:
            new_table.append(row)
    if len(new_table) == 1: # If there is only one table, return it as a list
        return header, new_table[0]
    return header, new_table

def table_by_columns(table, header, column_names):
    """ This function is perfect for accepting a large table and cutting it down to only the certain
    columns you want.
        Args:
            table (list of list of numeric vals): Data contained in the entire table
            header (list of obj): The list of strings that name each column
            column_names (list of obj): The subgroup of the header which you want to make a table of
        Returns:
            table_by_cols (list of list of numeric values) same table but only the necessary columns
    """
    attributes_list = []
    table_by_cols, table_by_cols_row = list(), list()
    for column_name in column_names:
        attributes_list.append(get_column(table,header,column_name))
    for i, attribute in enumerate(attributes_list[0]):
        for attributes in attributes_list:
            table_by_cols_row.append(attributes[i])
        table_by_cols.append(table_by_cols_row)
        table_by_cols_row = []
    return table_by_cols
def randints(noninclusive_max,num_integers,random_state=None):
    """ A simple recreation of the numpy function random.randint() but this version accepts either an integer
    for seeding or a value of None for seeding. 
    """
    np.random.seed(0)
    if random_state != None:
        np.random.seed(random_state)
    random_numbers = list(np.random.randint(0,noninclusive_max,size=num_integers))
    return random_numbers

def split_folds_to_train_test(folds):
    """ One fold is iterably assigned as the test fold while all others are combined iterably into
    the training folds. These two lists are returned as parallel with the 1 dimensional length
    equivalent to the different tests which can be run. 
        Args:
            folds (list of list of values) data placed into a certain number of folds 
        Returns:
            X_train_folds (list of list of values) contains lists of all data indexes for training, each list is
            an iteration for testing
            X_test_folds (list of list of values) contains lists of all data indexes for testing, each list is
            an iteration for testing
    """
    X_train_folds = []
    X_test_folds = []
    X_train_fold = []
    for fold_test in folds:
        X_test_folds.append(fold_test)
        for fold_train in folds:
            for instance in fold_train:
                if fold_train != X_test_folds[-1]: # Last item was the test fold, so do not add this
                    X_train_fold.append(instance)
        X_train_folds.append(X_train_fold)
        X_train_fold = []
    return X_train_folds, X_test_folds
def normal_round(n):
    """ Simple rounding function that accepts a number and correctly rounds it up
    or down as you would typically expect. 
    """
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)

def display_train_test(label, X_train, X_test, y_train, y_test):
    """
    """
    print("***", label, "***")
    print("train:")
    for i, _ in enumerate(X_train):
        print(X_train[i], "->", y_train[i])
    print("test:")
    for i, _ in enumerate(X_test):
        print(X_test[i], "->", y_test[i])
    print()

def randomize_in_place(alist, parallel_list=None,random_state=0):
    """Shuffles a list of data as well as an optional second list of data
    in parallel using the numpy random integer generator. 

        Args:
            alist (list of numeric vals): The list of data to be shuffled
            parallel_list (list of numeric vals): Another list of data to be shuffled in parallel
            random_state (int): number for seeding random number generator
    """
    np.random.seed(random_state)
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]

def group_by(table, header, groupby_col_name):
    """Computes subtables specific to various attribute values of the column requested. So, if the 
        column to be grouped by contains 3 different attribute options there will be 3 tables. 

        Args:
            table (list of numeric vals): The list of x values
            header (list of numeric vals): The list of y values
            group_by_col_name (obj): The string name of the column to be grouped by

        Returns:
            group_names (list of obj) string for each attribute used for grouping
            group_subtables (list of list of obj) list of tables, each should have only a single attribute in 
            the given groupby_col_name
    """
    groupby_col_index = header.index(groupby_col_name) # use this later
    if len(header) > 1:
        groupby_col = get_column(table, header, groupby_col_name)
    else:
        groupby_col = table
    group_names = sorted(list(set(groupby_col))) # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names] # e.g. [[], [], []]
    
    for row in table:
        if len(header) > 1:
            groupby_val = row[groupby_col_index] # e.g. this row's modelyear
        else:
            groupby_val = row
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(groupby_val)
        if len(header) > 1:
            group_subtables[groupby_val_subtable_index].append(row.copy()) # make a copy
        else:
            group_subtables[groupby_val_subtable_index].append(row) # make a copy
    
    return group_names, group_subtables
def compute_euclidian_distance(v1,v2,categorical=False):
    """Computes the euclidian distance between the two variables formed by their corresponding 

        Args:
            v1(list of numeric vals): The list of x values
            v2(list of numeric vals): The list of y values

        Returns:
            Value of the euclidian distance. 
        """
    if categorical:
        if v1 == v2:
            return 0
        else:
            return 1
    else:
        return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def get_column(table, header, col_name):
    """Grabs a singular column from a complex table with any number of attributes. 

        Args:
            table(list of list of vals): 2D table of continuous and categorical attributes
            header(list of obj): string names to describe the column attributes
            col_name(obj): string value for the singular column to be acquired

        Returns:
            col(list of obj): singular column of the table. 
        """
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col 

def get_frequencies(table, header, col_name):
    """
    Returns a list of unique values from a given list as a tuple with the corresponding frequencies
    of those unique values.
    """
    if len(header) > 1:
        col = get_column(table, header, col_name)
    else:
        col = table
    for element in col:
        col[col.index(element)] = str(element)
    col.sort() # inplace sort
    values = []
    counts = []
    for value in col:
        if value in values:
            counts[-1] += 1 # okay because col is sorted
        else: # haven't seen this value before
            values.append(value)
            counts.append(1)
    return values, counts # we can return multiple items

def discretize_doe_classification(y_column):
    """
    Converts a column holding continuous values for mileage per gallon to a categorical attribute.
    The list called conversions holds the data on how to convert mpg to a specific ranking of
    fuel economy based on DOE guidelines. 
    """
    conversions = [[10,"≥ 45", 45],[9,"37–44", 37],[8,"31–36", 31],[7,"27–30", 27],
                [6,"24–26", 24],[5,"20–23", 20],[4,"17–19", 17],[3,"15–16", 15],
                [2,"14", 14],[1,"≤ 13", 0]]
    y_discretized = []
    for y_value in y_column:
        for conversion in conversions:
            if y_value >= conversion[2]:
                y_discretized.append(conversion[0])
                break
    return y_discretized

def remove_column(table,header,column_name):
    """
    Removes a column from a table.
    """
    col_index = header.index(column_name)
    new_table = []
    for row in table:
        new_row = row[:col_index] + row[col_index+1:]
        new_table.append(new_row)
    return new_table

def normalize(column):
    new_column = []
    min = column[0]
    max = column[0]
    for value in column:
        if value > max:
            max = value
        if value < min:
            min = value
    range = max - min
    for value in column:
        # value = (value - min) / range
        new_column.append((value - min) / range)
    
    return new_column

def discretize(column):
    new_column = []
    for value in column:
        if value <= .1:
            value = 1
        elif value <= .2:
            value = 2
        elif value <= .3:
            value = 3
        elif value <= .4:
            value = 4
        elif value <= .5:
            value = 5
        elif value <= .6:
            value = 6
        elif value <= .7:
            value = 7
        elif value <= .8:
            value = 8
        elif value <= .9:
            value = 9
        else:
            value = 10
        new_column.append(value)

    return new_column

def discretizeY(column):
    new_column = []
    for value in column:
        if value == "R64":
            new_column.append(0)
        else:
            new_column.append(1)

    return new_column

def print_tree_helper(tree, rule, curr_att, attribute_names=None, class_name="class"):
    """ Recursive helper function for printing the rules of a tree
    Args:
        tree(nested list): current subtree being passed recursively
        rule(list): current rule being formed
        curr_att(string): current attribute of tree to keep track of rules
        attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
        class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        
    Returns:
        tree[1]: (string) final leaf node in tree, used to end function
    """

    info_type = tree[0]

    #checks if recursion needed
    if info_type == "Attribute":

        #gets current attribute and appends it to tree
        curr_att = tree[1]
        rule.append(tree[1])

        #loops trhough all values in current subtree
        for i in range(2, len(tree)):

            #reforms current rule based on the current attribute
            value_list = tree[i]
            curr_index = len(rule) - 1
            att_index = rule.index(curr_att)

            #deletets item from current rule index of current attribute is found
            while (curr_index != att_index):
                del rule[-1]
                curr_index -= 1

            # appends new value to rule
            rule.append("==")
            rule.append(value_list[1])
            rule.append("and")
            
            print_tree_helper(value_list[2], rule, curr_att, attribute_names, class_name)

    # leaf is found
    else: 

        # Prints out a rule
        print("If", end=" ")
        del rule[-1]
        for item in rule:
            if isinstance(item,str):
                if ("att" in item):
                    if (attribute_names != None):
                        print(attribute_names[int(item[3])], end= " ")
                    else:
                        print(item, end=" ")
                else:
                    print(item, end=" ")
            else:
                print(item, end=" ")
        
        print(", Then", class_name, "=", tree[1])
        # print()

        #returns last leaf to end function
        return tree[1]
    
