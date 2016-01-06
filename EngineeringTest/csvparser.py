"""CSV parser for parsing simple comma separated files

1. Design a CSV class to conveniently access individual cells of the csv.
The parser should parse a given text file and return an instantiation of the CSV
class.

2. Additionally we want to detect and store in the CSV class the type info-
rmation for each of the columns in the CSV file

>>> chau,30,94110
"String", "Numeric", "Numeric"

Edge case: '94110' should return a string, such that regex should handle \'94105\'
            Handled by not removing extra quotation marks if string does not
            contain a comma.
"""

import os
import sys
import re

# Regular Expression for handling commas in strings
csv_pat = re.compile(r"""
    \s*                # Any whitespace.
    (                  # Start capturing here.
      [^,"']+?         # Either a series of non-comma non-quote characters.
      |                # OR
      "(?:             # A double-quote followed by a string of characters...
          [^"\\]|\\.   # That are either non-quotes or escaped...
       )*              # ...repeated any number of times.
      "                # Followed by a closing double-quote.
      |                # OR
      '(?:             # Same as above, for single quotes.
          [^'\\]|\\.
       )*'              
    )                  # Done capturing.
    \s*                # Allow arbitrary space before the comma.
    (?:,|$)            # Followed by a comma or the end of a string.
    """, re.VERBOSE)


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
        self.csv_files = []  # for storing filenames

        # change current working directory to filepath
        os.chdir(self.filepath)

        # add files in directory to csv_files
        suffix = ('.CSV', '.csv')
        for file in os.listdir(self.filepath):
            if file.endswith(suffix):
                self.csv_files.append(file)

        # create dictionary for storing all data from all files in directory
        self.all_data = {}
        self.all_types = {}

        return


    def read(self):
        """Reads all files in folder into memory."""

        # self.filenames = {}    

        for filename in self.csv_files:
            print "FILE:", filename


            # Read CSV
            self.curr_data = [line.rstrip('\n').rstrip('\r') for line in open(filename, 'rb')]
            self.clean_data = [csv_pat.findall(item) for item in self.curr_data]
            # clean off extra quotations surrounding strings with commas
            for row in self.clean_data:
                # if filename == 'example2.csv':
                #     import pdb; pdb.set_trace()
                for i, item in enumerate(row):
                    if '"' in item and ',' in item:
                        row[i] = item.strip('"')
                    elif "'" in item and ',' in item:
                        row[i] = item.strip("'")
            # add current clean data to dictionary by filename
            self.all_data[filename] = self.clean_data

        return


    def types(self, filename):
        """Return the data type information for all cells in a file."""
        types_by_row = []

        for j, row in enumerate(self.all_data[filename]):
            curr_row = 'List('
            for i, item in enumerate(row):
                if item.isdigit():
                    curr_row += '"Numeric",'
                else:
                    curr_row += '"String",'
                if i == len(row) - 1:
                    curr_row = curr_row[:-1]
                    curr_row += ')'
            print "ROW %d: %s" % (j, curr_row)
            types_by_row.append(curr_row)
        
        # add types to dictionary
        self.all_types[filename] = types_by_row
        
        return


if __name__ == "__main__":
    my_dir = "."

    print "Creating CSV Parser for files in %s ..." % os.path.abspath(my_dir)
    CSV_reader = CSVParse(my_dir)

    print
    print "Reading files in current working directory ..."
    CSV_reader.read()

    test_files = ['example.csv', 'example2.csv']
    for test_file in test_files:
        print
        print "FILE:", test_file
        print "File information by row ... "
        for i, row in enumerate(CSV_reader.all_data[test_file]):
            print "ROW %d: %s" % (i, row)
        print
        print "Reading data type information by row ..."
        CSV_reader.types(test_file)


