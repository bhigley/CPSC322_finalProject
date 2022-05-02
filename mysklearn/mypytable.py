from mysklearn import myutils
##############################################
# Programmer: Matthew Moore
# Class: CptS 322-01, Spring 2022
# Programming Assignment #4
# 2/9/2022
# 
# 
# Description: This program is the blue print for the object of a python table which essentially
# mimics the functionality of a Pandas Data Frame in many ways. I built the algorithm to manipulate
# a table of data in a lot of ways, the most complex being the inner join and the full outer join.
##############################################

import copy
import csv
import logging
import statistics


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

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

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
        print(self.data)
        single_column = []
        for row in self.data:
            try:
                single_column.append(row[self.column_names.index(col_identifier)])
            except ValueError:
                pass

        return single_column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for element in row:
                row_index = self.data.index(row)
                attribute_index = self.data[row_index].index(element)
                try: 
                    self.data[row_index][attribute_index] = float(self.data[row_index][attribute_index])
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        updated_table = []

        for row in self.data:
            row_index = self.data.index(row)
            try:
                if row_index not in row_indexes_to_drop:
                    updated_table.append(self.data[row_index])
            except IndexError:
                print("Index Error: Out of range value for list")

        self.data = updated_table

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
        with open(filename, 'r') as file:
            csvreader = csv.reader(file)
            self.column_names = next(csvreader)
            for row in csvreader:
                self.data.append(row)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename,'w',newline='') as file:
            the_writer = csv.writer(file)
            the_writer.writerow(self.column_names)
            for row in self.data:
                the_writer.writerow(row)

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
        key_column_table = []
        key_column_row = []
        key_column_found = []
        dup_indexes = []
        # Builds a table of only the key values
        for row in self.data:
            for column_name in key_column_names:
                key_column_row.append(row[self.column_names.index(column_name)])
            key_column_table.append(key_column_row)
            key_column_row = []
        # Appends repeat indexes to a new list
        curr_index = 0
        for row in key_column_table:
            if row in key_column_found:
                dup_indexes.append(curr_index)
            else: # This is a case of the first occurance
                key_column_found.append(row)
            curr_index += 1

        return dup_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        drop_indexes = []
        curr_index = 0
        for row in self.data:
            for attribute in row:
                if attribute == "NA":
                    drop_indexes.append(curr_index)
            curr_index += 1
        
        self.drop_rows(drop_indexes)


    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        gross_sum = 0
        gross_count = 0
        for row_to_count in self.data:
            attribute_to_count = row_to_count[self.column_names.index(col_name)]
            if type(attribute_to_count) == type(1.0):
                gross_sum = gross_sum + attribute_to_count
                gross_count += 1
            elif attribute_to_count == None:
                pass # Fixing these
            else:
                exit # Not a continuous row so get out

        average_of_column = gross_sum / gross_count
        fixed_table = []
        fixed_row = []

        for row in self.data:
            attribute_to_fix = row[self.column_names.index(col_name)]
            for attribute in row:
                if attribute_to_fix == attribute:
                    if attribute == "NA":
                        fixed_row.append(average_of_column)
                    else:
                        fixed_row.append(attribute)
                else:        
                    fixed_row.append(attribute)
            fixed_table.append(fixed_row)
            fixed_row = []
        
        self.data = fixed_table


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        header = ["attribute","min","max","mid","avg","median"]
        the_data = []
        row_in_the_data = []
        for stat_column in col_names:
            nums_to_analyze = self.get_column(stat_column)
            if nums_to_analyze:
                row_in_the_data.append(stat_column)
                row_in_the_data.append(min(nums_to_analyze))
                row_in_the_data.append(max(nums_to_analyze))
                row_in_the_data.append((max(nums_to_analyze) + min(nums_to_analyze)) / 2)
                row_in_the_data.append(statistics.mean(nums_to_analyze))
                row_in_the_data.append(statistics.median(nums_to_analyze))
                the_data.append(row_in_the_data)
                row_in_the_data = []
        return MyPyTable(header,the_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        other_not_key_header = []
        # Creates header list for columns not in the key for other table
        for other_column_name in other_table.column_names:
            if other_column_name not in key_column_names:
                other_not_key_header.append(other_column_name)

        new_table = []
        new_row = []
        main_key_attributes = []
        main_not_key_attributes = []
        other_key_attributes = []
        other_not_key_attributes = []
        for main_row in self.data:
            for key_column in key_column_names: # Creates row of only key attributes of table 1
                main_key_attributes.append(main_row[self.column_names.index(key_column)])
            for main_element in main_row: # Creates row of only non-key attributes of table 1
                if main_element not in main_key_attributes:
                    main_not_key_attributes.append(main_element)
            for other_row in other_table.data:
                for other_key_column in key_column_names: # Creates row of only key attributes of table 2
                    other_key_attributes.append(other_row[other_table.column_names.index(other_key_column)])
                for other_element in other_row: # Creates row of only non-key attributes of table 2
                    if other_element not in other_key_attributes:
                        other_not_key_attributes.append(other_element)
                if main_key_attributes == other_key_attributes: # Joins columns if key values match
                    new_table.append(main_row + other_not_key_attributes)
                other_key_attributes = []
                other_not_key_attributes = []
            main_key_attributes = []
            main_not_key_attributes = []
            

        return MyPyTable(self.column_names + other_not_key_header, new_table)

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

        other_not_key_header = []
        # Creates header list for columns not in the key for other table
        for other_column_name in other_table.column_names:
            if other_column_name not in key_column_names:
                other_not_key_header.append(other_column_name)

        new_table = []
        main_key_attributes = []
        main_not_key_attributes = []
        other_key_attributes = []
        other_not_key_attributes = []
        main_rows_inner = []
        other_rows_inner = []
        empty_list = []
        for number in range(len(other_not_key_header)):
            empty_list.append("NA")

        for main_row in self.data:
            for key_column in key_column_names: # Creates row of only key attributes of table 1
                main_key_attributes.append(main_row[self.column_names.index(key_column)])
            for main_element in main_row: # Creates row of only non-key attributes of table 1
                if main_element not in main_key_attributes:
                    main_not_key_attributes.append(main_element)
            for other_row in other_table.data:
                for other_key_column in key_column_names: # Creates row of only key attributes of table 2
                    other_key_attributes.append(other_row[other_table.column_names.index(other_key_column)])
                for other_element in other_row: # Creates row of only non-key attributes of table 2
                    if other_element not in other_key_attributes:
                        other_not_key_attributes.append(other_element)
                if main_key_attributes == other_key_attributes: # Joins columns if key values match
                    new_table.append(main_row + other_not_key_attributes)
                    main_rows_inner.append(main_row)
                    other_rows_inner.append(other_row)
                other_key_attributes = []
                other_not_key_attributes = []
            if main_row not in main_rows_inner:
                new_table.append(main_row + empty_list)
            main_key_attributes = []
            main_not_key_attributes = []

        empty_list.clear()
        main_key_attributes.clear()
        other_row_outer = []
        for other_row in other_table.data:
            for main_header in self.column_names: # Building an empty portion of the row to mimic the left table row
                if main_header in key_column_names:
                        empty_list.append(other_row[other_table.column_names.index(main_header)])
                else:
                    empty_list.append("NA")
            if other_row not in other_rows_inner:
                for other_header in other_table.column_names: # Building the other portion of the row from the right table (inner table)
                    if other_header not in key_column_names: # Does not include the key attributes, as they are already included
                        other_row_outer.append(other_row[other_table.column_names.index(other_header)])
                new_table.append(empty_list + other_row_outer)
            empty_list.clear()
            other_row_outer.clear()

        return MyPyTable(self.column_names + other_not_key_header, new_table)


