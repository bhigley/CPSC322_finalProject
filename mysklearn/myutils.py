##############################################
# Programmer: Matthew Moore
# Class: CptS 322-01, Spring 2022
# Programming Assignment #4
# 2/9/2022
# 
# 
# Description: This program holds all of the utility functions utilized in the entire project.
# these functions are basic, general purpose functions to be used to minimize code in our classifiers.
##############################################
import numpy as np

def compute_euclidian_distance(v1,v2):
    """Computes the euclidian distance between the two variables formed by their corresponding 

        Args:
            v1(list of numeric vals): The list of x values
            v2(list of numeric vals): The list of y values

        Returns:
            Value of the euclidian distance. 
        """
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

def normalize(column):
    # i = lower_index
    # for i in range(higher_index): # iterates trhough each attribute
    #     min = table[0][i]
    #     max = table[0][i]
    #     for row in table:
    #         if row[i] > max:
    #             max = row[i]
    #         if row[i] < min:
    #             min = row[i]
    #     ran = max - min
    #     for row in table:
    #         row[i] = (row[i] - min) / ran
    #     ran = 0
    # print(table[5])
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

    print(new_column)

    return new_column



