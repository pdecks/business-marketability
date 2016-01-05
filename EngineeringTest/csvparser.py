"""CSV parser for parsing simple comma separated files

1. Design a CSV class to conveniently access individual cells of the csv.
The parser should parse a given text file and return an instantiation of the CSV
class.

2. Additionally we want to detect and store in the CSV class the type info-
rmation for each of the columns in the CSV file

>>> chau,30,94110
“String”, “Numeric”, “Numeric”
"""

# import numpy as numpy
import os
import sys

class CSVParse(object):
    """Open CSV file(s) and store info using lists or numpy arrays.
    
    The class takes the following arguments:

    CSVParse(filepath, skip_rows=0)

    filepath: string of the folder location containing .CSV file(s)

    skip_rows: specified rows at the top of the file to skip
    """

    def __init__(self, filepath, skip_rows=0):
        self.skip_rows = skip_rows
        self.filepath = str(filepath)
        self.csv_files = []

        # change current working directory to filepath
        os.chdir(self.filepath)

        # add files in directory to csv_files
        suffix = ('.CSV', '.csv')
        for file in os.listdir(self.filepath):
            if file.endswith(suffix):
                self.csv_files.append(file)

        # create array for storing all data from all files in directory
        self.all_data = [0] * len(self.csv_files)

