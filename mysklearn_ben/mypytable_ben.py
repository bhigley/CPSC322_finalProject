import copy
import csv

from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        row_count = 0
        item_count = 0
        row_count = len(self.data)
        item_count = len(self.data[0])

        return row_count, item_count

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        print("test")
        column_index = 0
        for item in self.column_names:
            if col_identifier == item:
                break
            column_index += 1
        print(column_index)
        column_list = []
        for row in self.data: # deep copy without import
            if include_missing_values is True:
                column_list.append(row[column_index])
            else:
                if row[column_index] != "NA":
                    column_list.append(row[column_index])
        return column_list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        row_iterator = 0 # used to index into table at row
        item_iterator = 0 # used to index into table at column
        for row in self.data:
            for item in row:
                try:
                    item = float(item)
                    self.data[row_iterator][item_iterator] = item
                except:
                    pass
                item_iterator += 1
            item_iterator = 0
            row_iterator += 1

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        for item in reversed(row_indexes_to_drop):
            self.data.pop(item)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """

        with open(filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            self.column_names = next(csv_reader)
            for line in csv_reader:
                self.data.append(line)
        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(self.column_names)
            for row in self.data:
                csv_writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        index = 0
        column_indexs = []
        # build of list of row indexs to check
        for column_iterator in self.column_names:
            for item in key_column_names:
                if item == column_iterator:
                    column_indexs.append(index)
            index += 1

        duplicate_indexs = []
        duplicate = True
        iterator = 0
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                for item in column_indexs:
                    if self.data[i][item] != self.data[j][item]:
                        duplicate = False
                if duplicate is True:
                    duplicate_indexs.append(j)
                duplicate = True
        for item in duplicate_indexs:
            if duplicate_indexs.count(item) > 1:
                duplicate_indexs.pop(iterator)
            iterator += 1


        return duplicate_indexs

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        row_iterator = 0
        remove_list = []
        for row in self.data:
            for item in row:
                if item == "NA":
                    remove_list.append(row_iterator)
            row_iterator += 1
        for item in reversed(remove_list):
            self.data.pop(item)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        sum_val = 0
        average = 0
        table_copy = copy.deepcopy(self) # needed to compute average
        table_copy.remove_rows_with_missing_values()
        # get column index
        column_index = 0
        for item in self.column_names:
            if col_name == item:
                break
            column_index += 1
        for row in table_copy.data:
            sum_val = sum_val + row[column_index]

        average = sum_val / len(table_copy.data)

        for row in self.data:
            if row[column_index] == "NA":
                row[column_index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        new_columns = ["attribute", "min", "max", "mid", "avg", "median"]
        cleaned_row = []
        new_row = []
        new_table = MyPyTable(new_columns)
        for item in col_names:
            if len(self.get_column(item)) > 0:
                cleaned_row = self.get_column(item, False) # gets the attribtues without NA values
                new_row = new_row + [item] # adding the attribute name to the new row
                new_row = new_row + [compute_min(cleaned_row)]
                new_row = new_row + [compute_max(cleaned_row)]
                new_row = new_row + [compute_mid(cleaned_row)]
                new_row = new_row + [compute_avg(cleaned_row)]
                new_row = new_row + [compute_median(cleaned_row)]
                new_table.data.append(new_row)
                new_row = [] # clear row for next attribtue

        return new_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        control = True
        i = 0
        j = 1 # used if key indexes aren't in the same order
        indexes_match = True # used to check if key_column names are in the same index for each table
        matched_indexes = 0 # used to check if rows match at multiple keys
        new_column_names = []
        new_row = []
        new_table = MyPyTable()
        iterator = 0 # used to iterate through key indexes
        self_indexes = [] # holds the key value indexs of the original table
        other_indexes = [] # holds the key value indexes of the other table
        column_index = 0

        # get the column indexes
        for item in self.column_names:
            for key in key_column_names:
                if key == item:
                    self_indexes.append(column_index)
            column_index += 1
        column_index = 0 # reset column index for other table
        for item in other_table.column_names:
            for key in key_column_names:
                if key == item:
                    other_indexes.append(column_index)
            column_index += 1
        # build up new table headers
        for item in self.column_names:
            new_column_names.append(item)
        for item in other_table.column_names:
            new_column_names.append(item)
        # delete duplicate column names
        for item in new_column_names:
            if new_column_names.count(item) > 1:
                new_column_names.pop(iterator)
            iterator += 1
        # check that column names match indexes
        if (len(self_indexes) > 0 and len(other_indexes)): # avoiding index out of range
            if self.column_names[self_indexes[0]] != other_table.column_names[other_indexes[0]]:
                indexes_match = False
        # add new headers to new table
        new_table.column_names = new_column_names
        # reset iterator
        iterator = 0

        for self_row in self.data:
            for other_row in other_table.data:
                for i in range(len(self_indexes)): # checks that all keys are matches
                    if indexes_match is True:
                        if self_row[self_indexes[i]] == other_row[other_indexes[i]]:
                            matched_indexes += 1
                        if matched_indexes < len(self_indexes):
                            control = False
                        else:
                            control = True
                    else:
                        if self_row[self_indexes[i]] == other_row[other_indexes[j]]:
                            matched_indexes += 1
                        j = 0 # set to zero to pull first index on next loop
                        if matched_indexes < len(self_indexes):
                            control = False
                        else:
                            control = True
                j = 1 # j reset to 1 for next loop
                if control is True:
                    for item in self_row: # problem is here
                        new_row = new_row + [item]
                    for i in range(len(other_row)):
                        if other_indexes.count(i) < 1: # only adds columns that aren't keys
                            new_row = new_row + [other_row[i]]
                        i += 1
                matched_indexes = 0
                control = True # reset it in the case it was not a match\
                if len(new_row) > 0: # checks that list is filled and needs to be added
                    new_table.data.append(new_row)
                new_row = []
        iterator += 1

        return new_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        control = True
        row_added = False
        i = 0
        j = 1 # used if key indexes aren't in the same order
        indexes_match = True # used to check if key_column names are in the same index for each table
        matched_indexes = 0 # used to check if rows match at multiple keys
        new_column_names = []
        new_row = []
        new_table = MyPyTable()
        iterator = 0 # used to iterate through key indexes
        self_indexes = [] # holds the key value indexs of the original table
        other_indexes = [] # holds the key value indexes of the other table
        column_index = 0

        # the same process to collect column indexes as inner join
        # get the column indexes
        for item in self.column_names:
            for key in key_column_names:
                if key == item:
                    self_indexes.append(column_index)
            column_index += 1
        column_index = 0 # reset column index for other table
        for item in other_table.column_names:
            for key in key_column_names:
                if key == item:
                    other_indexes.append(column_index)
            column_index += 1
        # build up new table headers
        for item in self.column_names:
            new_column_names.append(item)
        for item in other_table.column_names:
            if key_column_names.count(item) < 1:
                new_column_names.append(item)
        # check that column names match indexes
        if (len(self_indexes) > 0 and len(other_indexes)): # avoiding index out of range
            if self.column_names[self_indexes[0]] != other_table.column_names[other_indexes[0]]:
                indexes_match = False
        # add new headers to new table
        new_table.column_names = new_column_names
        # reset iterator
        iterator = 0

        # starts with inner join
        for self_row in self.data:
            for other_row in other_table.data:
                for i in range(len(self_indexes)): # checks that all keys are matches
                    if indexes_match is True:
                        if self_row[self_indexes[i]] == other_row[other_indexes[i]]:
                            matched_indexes += 1
                        if matched_indexes < len(self_indexes):
                            control = False
                        else:
                            control = True
                    else:
                        if self_row[self_indexes[i]] == other_row[other_indexes[j]]:
                            matched_indexes += 1
                        j = 0 # set to zero to pull first index on next loop
                        if matched_indexes < len(self_indexes):
                            control = False
                        else:
                            control = True
                j = 1 # j reset to 1 for next loop
                if control is True:
                    for item in self_row:
                        new_row = new_row + [item]
                    for i in range(len(other_row)):
                        if other_indexes.count(i) < 1: # only adds columns that aren't keys
                            new_row = new_row + [other_row[i]]
                        i += 1
                matched_indexes = 0
                control = True # reset it in the case it was not a match\
                if len(new_row) > 0: # checks that list is filled and needs to be added
                    new_table.data.append(new_row)
                    row_added = True
                new_row = []
            if row_added is True:
                row_added = False # reset and continue to next row
            else: # perform outer join procedure
                new_row = self_row
                for i in range(len(other_row)):
                    if other_indexes.count(i) < 1: # only adds columns that aren't keys
                        new_row = new_row + ["NA"]
                    i += 1
                new_table.data.append(new_row)
                new_row = []
        iterator += 1

        # perform outer join on non-matched other table rows
        matched_indexes = 0
        for other_row in other_table.data:
            for row in new_table.data:
                for i in range(len(self_indexes)): # checks that all keys are matches
                    if indexes_match is True:
                        if row[self_indexes[i]] == other_row[other_indexes[i]]:
                            matched_indexes += 1
                        if matched_indexes < len(self_indexes):
                            control = False
                        else:
                            control = True
                            break
                    else:
                        if row[self_indexes[i]] == other_row[other_indexes[j]]:
                            matched_indexes += 1
                        j = 0 # set to zero to pull first index on next loop
                        if matched_indexes < len(self_indexes):
                            control = False
                        else:
                            control = True
                            break
                j = 1 # reset for next iteration
                matched_indexes = 0
                if control is True: # other_row has a match
                    break

            if control is False: # no match found
                new_row = []
                j = 1
                i = 0
                index_iterator = 0
                for i in range(len(self.column_names)):
                    if self_indexes.count(i) < 1: # only adds columns that aren't keys
                        new_row = new_row + ["NA"]
                    else: # item is a key, add key value from other_table
                        if indexes_match is True:
                            new_row = new_row + [other_row[other_indexes[index_iterator]]]
                            index_iterator += 1
                        else: # indexes dont match
                            new_row = new_row + [other_row[other_indexes[j]]]
                            j  = j - 1
                for i in range(len(other_row)):
                    if other_indexes.count(i) < 1: # only adds columns that aren't keys
                        new_row = new_row + [other_row[i]]
                    i += 1
                new_table.data.append(new_row)
                new_row = []
            control = False # reset to no match found for next iteration

        return new_table

def compute_min(data):
    """Finds the minimum value in a list

        Args:
            data(list of floats) a list that holds each value of a column

        Returns
            int: that holds the minimum value in a list
        """
    min_val = data[0]
    for item in data:
        if item < min_val:
            min_val = item

    return min_val

def compute_max(data):
    """Computes the maximum value in a list

        Args:
            data(list of floats) a list that holds each value of a column

        Returns
            int: that holds the maximum value in a list
        """
    max_val = data[0]
    for item in data:
        if item > max_val:
            max_val = item

    return max_val

def compute_mid(data):
    """Finds the mid value in a list

        Args:
            data(list of floats) a list that holds each value of a column

        Returns
            int: that holds the mid value in a list
        """
    max_val = compute_max(data)
    min_val = compute_min(data)
    mid = (max_val + min_val) / 2

    return mid

def compute_avg(data):
    """Computes the average value of a list

        Args:
            data(list of floats) a list that holds each value of a column

        Returns
            float: that holds the average value for a list
        """
    sum_val = 0
    average = 0
    for item in data:
        sum_val = sum_val + item
    average = sum_val / len(data)

    return average

def compute_median(data):
    """Computes the median value in a list

        Args:
            data(list of floats) a list that holds each value of a column

        Returns
            int: that holds the median value in a list
        """
    data.sort()
    if len(data) % 2 == 0: # even number of items
        median = ((data[len(data)//2 - 1] + data[len(data)//2])) / 2
    else:
        median = data[len(data) // 2]

    return median
