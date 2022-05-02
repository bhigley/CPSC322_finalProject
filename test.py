import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable

import mysklearn.myutils
import mysklearn.myutils as myutils


table = MyPyTable()
table.load_from_file("input_data/cbb.csv")

myutils.normalize(table.get_column("TOR"))