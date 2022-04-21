import numpy as np
import math

def discretize(value):
    classes = ["2ND", "Champions", "E8", "F4", "R32", "R64", "R68", "S16"]
    index = classes.index(value)
    if index < 4 or index == 7:
        return 0
    else:
        return 1

def partition_instances(instances, split_attribute, attribute_domains, header):
    """Splits the instances into partitions based on an attribute

        Args:
            instances : available instances that could be split
            split_attribute: the attribute that will be split on
            attribute_domains: dictionary holding possible values for an attribute
            header : list of string holding attribute names

        Returns:
            partitions : list of instances according to a specific attribute domain

        Notes:
            needed for recursive descent
        """
    # lets use a dictionary
    partitions = {} # key (string): value (subtable)
    att_index = header.index(split_attribute) # e.g. 0 for level
    att_domain = attribute_domains[header[att_index]] # e.g. ["Junior", "Mid", "Senior"]
    for att_value in att_domain:
        partitions[att_value] = []
        # TASK: finish
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def all_same_class(attribute_partition):
    """Checks if all remaining attributes are same class for a case 1

        Args:
            attribute_partition : holds the remaining values

        Returns:
            true or false

        Notes:
            needed for determining the case
        """
    label = attribute_partition[0][-1]
    for attribute in attribute_partition:
        if attribute[-1] != label:
            return False
    
    return True

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

def tdidt(current_instances, available_attributes, attribute_domains, header):
    """Generatest the decision tree
        Args:
            current_instances : instances that haven't been made into a rule
            available_attributes : attributes that can still be split on

        Returns:
            the tree generated
        Notes:
            is a recursive function
        """
    attribute = select_attribute(current_instances, available_attributes, header)
    available_attributes.remove(attribute)
    tree = ["Attribute", attribute]
    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, attribute, attribute_domains, header)
    # for each partition, repeat unless one of the following occurs (base case)
    skip = False
    for att_value, att_partition in partitions.items():
        values_subtree = ["Value", att_value]
    #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            leaf_node = ["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)]
            values_subtree.append(leaf_node)
    #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif (len(att_partition) > 0 and len(available_attributes) == 0):
            label = majority_vote(att_partition)
            leaf_node = ["Leaf", label, len(att_partition), len(current_instances)]
            values_subtree.append(leaf_node)
    #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            skip = True
            tree = ["Leaf", majority_vote(current_instances), len(current_instances), len(current_instances)] # need to fix len
        else: # previous conditions are all false... recurse!
            subtree = tdidt(att_partition, available_attributes.copy(), attribute_domains, header)
            values_subtree.append(subtree)
            # note the copy
        if skip == False:
            tree.append(values_subtree)
    return tree


def compute_euclidean_distance(v1, v2):
    """ computes euclidean distance for paralel lists passed in
    Args:
        vl: (list) list of values
        v2: (list) paralel list of values
    Returns:
        dist: euclidean distance of passed in lists
    """
    
    assert len(v1) == len(v2)

    dist = []
    for i in range(len(v1)):
        if (isinstance(v1[i],str) or isinstance(v2[i],str)):
            if (v1[i] == v2[i]):
                    dist.append(0)
            else:
                dist.append(1)
        else:
            dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
            return dist
    return dist

def get_frequencies(col):
    """ gets frequencies for the passed in col_name and returns parallel arrays
    with the values in the collumns and the counts
    Args:
        col: (list) column name of frequencies to find
    Returns:
        values: (list) values in col_name
        counts: (list) paralel list to values list of frequency counts
    """

    col.sort() # inplace
    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts 


def get_column(table, col_index):
    """ gets the column from a passed in col_name
    Args:
        table: (list of lists) table of data to get column from
        col_index: index of column in table
    Returns:
        col: (list) column wanted
    """
    col = []
    for row in table: 
        # ignore missing values ("NA")
        if row[col_index] != "NA":
            col.append(row[col_index]) # was in if statement
    return col

def group_by(table, col_index):
    """ groups the table by the passed in col_index
    Args:
        table: (list of lists) table of data to get column from
        col_index: index of column in table
    Returns:
        group_names: (list) list of group label names
        group_subtables: (list of lists) 2d list of each group subtable
    """
    col = get_column(table, col_index)

    # get a list of unique values for the column
    group_names = sorted(list(set(col))) # 75, 76, 77
    group_subtables = [[] for _ in group_names] # [[], [], []]

    # walk through each row and assign it to the appropriate
    # subtable based on its group by value (model year)
    for row in table:
        group_value = row[col_index]
        # which group_subtable??
        group_index = group_names.index(group_value)
        group_subtables[group_index].append(row.copy()) # shallow copy

    return group_names, group_subtables

def select_attribute(instances, available_attributes, header):

    """ Selects attributes using entropy
    Args:
        instances: (list of lists) table of data
        available_attributes: (list) list of available attributes to split on
        header: (list) header of attributes
    Returns:
        attribute: (string) attribute selected to split on
    """
    
    table_size = len(instances)
    e_new_list = []

    # loops through each available attribute and groups data by each attribute
    for item in available_attributes : 
        group_names, group_subtables = group_by(instances, header.index(item))
        e_value_list = []
        num_values = []

        # loops through the group subtable and further groups by class name
        for j in range(len(group_subtables)):
            curr_group = group_subtables[j]
            num_attributes = len(curr_group)
            num_values.append(num_attributes)
            class_names, class_subtables = group_by(curr_group, len(curr_group[0])-1)
            e_value = 0

            #checks for empty partition for log base 2 of 0 calculations
            if (len(class_subtables) == 1):
                    e_value = 0
            else :
                #loops through each group bay attribute class and calculates the entropy
                for k in range(len(class_subtables)):
                    class_num = len(class_subtables[k]) / num_attributes
                    e_value -= (class_num) * (math.log2(class_num))
            e_value_list.append(e_value)
        
        e_new = 0

        #calculates e_new for each attribute 
        for l in range (len(e_value_list)):
            e_new += e_value_list[l] * (num_values[l]/ table_size)
        e_new_list.append(e_new)

    #finds attribute with minimum entropy and selects that attribute
    min_entropy = min(e_new_list)
    min_index = e_new_list.index(min_entropy)
    attribute = available_attributes[min_index]

    return attribute


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


def group_by_multiple_atts(indexes, y):
    """Groups data based on multiple attributes

        Args:
            indexes: list of ints
            y :v alues

        Returns:
            grouped indexes

        Notes:
            needed for naive predict
        """
    groups = []
    group_indexes = []
    for item in y:
        if item not in groups:
            groups.append(item)
            group_indexes.append([])
    groups.sort()
    i = 0
    for item in y:
        for j in range(len(groups)):
            if item == groups[j]:
                group_indexes[j].append(indexes[i])
        i += 1
    return group_indexes

def find_max(values):
    """FInds the max of a list of values

        Args:
            values : list of floats

        Returns:
            max_index : holds the index of the max_value found

        Notes:
            used in naive predict
        """
    max_val = 0
    max_index = 0
    for i in range(len(values)):
        if values[i] > max_val:
            max_val = values[i]
            max_index = i
    return max_index

def get_num_instances(name, indexes, column):
    """Gets the number of instances for a value

        Args:
            name : holds attribute value
            indexes : holds the indexes to check
            column : holds the values for a given attribtue


        Returns:
            count : number of instances

        Notes:
            needed for naive predict
        """
    count = 0
    for index in indexes:
        if column[index] == name:
            count += 1
    return count

def adjust_X_test(X_test, table):
    """Adjusts the X_test values so that they can index into the posteriors correctly

        Args:
            X_test : holds the values that need to be adjusted
            table : holds the data that is used to adjust the values

        Returns:
            adjusted_X : holds the adjusted values for X_test

        Notes:
            needed for naive predict
        """
    attribute_values = []
    adjusted_X = []
    attribute_count = 0 # used to tell which attribute category you are in
    for i in range(len(table[0])):
        values, counts = get_frequencies(get_column(table, i))
        attribute_values.append(values)
    i = 0
    for item in X_test:
        new_X_item = []
        for val in item:
            for item in attribute_values[attribute_count]:
                if val == item:
                    score = i # score assigned to the index value from the attribute item
                    for i in range(attribute_count): # loops the number of attribtue you have already checked
                        score = score + len(attribute_values[i])
                    new_X_item.append(score)
                    attribute_count += 1
                i += 1
            i = 0
        adjusted_X.append(new_X_item)
        attribute_count = 0

    return adjusted_X

def convert(predictions):
    """Converts the numeric predictions to A or H for bball dataset

        Args:
            predictions : list of int

        Returns:
            predictions_new : list of chars

        Notes:
            needed for consistent printing
        """
    predictions_new = []
    for item in predictions:
        if item == 0:
            predictions_new.append("A")
        else:
            predictions_new.append("H")
    return predictions_new

