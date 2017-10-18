

import csv
import numpy as np
import sqlite3
from itertools import islice, tee
from datetime import datetime

class ChunkDictReader():
    """
    A class to gradually read chunks of information from a large csv file (using a dictionary reader - the file must have fieldnames at the top). Ideally to be used for machine learning problems where the dataset is very large, and doesn't fit comfortably into memory.
    """
    def __init__(self, filename, split=0.8):
        """
        Takes in the filename to open, and the fieldnames that it wants extracted from the file.
        """
        self.filename = filename

        # count number of lines in document to make the train/val split
        self.num_lines = sum(1 for line in open(filename, "r"))

        self.split = split * self.num_lines

        # initializing the csv reader
        self.fh = open(filename, "r")
        self.all_data_reader = csv.DictReader(self.fh, delimiter=",")

        # make two copies of the reader to split into training and validation
        self.reader_train, self.reader_val = tee(self.all_data_reader, 2)

        # create "readers" for training and cross-validation data
        self.reader_train = islice(self.reader_train, 0, self.split)
        self.reader_val = islice(self.reader_val, self.split, self.num_lines)


    def next(self, chunksize):
        """
        Method that loads the next (chunksize) lines and returns them as a list of dictionaries (with fieldnames).
        Returns None when it is done reading
        """
        lineiter = islice(self.reader_train, 0, chunksize)
        lines = [line for line in lineiter]
        return lines

    def next_val(self, chunksize):
        """
        Method that loads the next (chunksize) lines of the cross-validation set and returns them as a list of dictionaries (with fieldnames)
        """
        lineiter = islice(self.reader_val, 0, chunksize)
        lines = [line for line in lineiter]
        return lines


class ChunkDBReader():
    """
    Performs the same functions as the ChunkDictReader but reading from a database. Returns a list of dictionaries with all the data under their corresponding column names (like a csv.DictReader).
    """
    def __init__(self, dbfile, table_name, split=0.8):

        self.table_name = table_name
        self.conn = sqlite3.connect(dbfile)
        self.cur = self.conn.cursor()

        self.num_lines = self.cur.execute("SELECT count(*) FROM " + self.table_name).next()[0]
        self.split = split * self.num_lines

        self.train_counter = 0
        self.val_counter = int(self.split)

        # get the column names (fieldnames)
        self.fieldnames = []
        fields = self.cur.execute("PRAGMA table_info(" + self.table_name + ")")
        for field in fields:
            # name is in the second position of the tuple
            self.fieldnames.append(field[1])

    def next(self, chunksize):

        # training examples left
        diff = self.split - self.train_counter

        # if chunksize is greater than the examples left, return the examples left. If there are no more examples left, return an empty list
        if diff < chunksize:
            chunksize = diff
        elif diff <= 0:
            chunksize = 0

        lines = self.cur.execute("SELECT * FROM " + self.table_name + " ORDER BY rowid LIMIT ? OFFSET ?", (int(chunksize), int(self.train_counter)))

        lines = self._convert_to_dict(lines)

        self.train_counter += chunksize
        return lines

    def next_val(self, chunksize):

        lines = self.cur.execute("SELECT * FROM " + self.table_name + " ORDER BY rowid LIMIT ? OFFSET ?", (int(chunksize), int(self.val_counter)))
        lines = self._convert_to_dict(lines)

        self.val_counter += chunksize

        return lines

    def _convert_to_dict(self, iterator):
        """
        Maps the data in the :iterator: to a list of dictionaries, each containing the names in :self.fieldnames:. The names should be ordered in the same order as items in the iterator.
        """
        lines = []

        for record in iterator:
            new_rec = {}
            for idx, name in enumerate(self.fieldnames):
                new_rec[name] = record[idx]
            lines.append(new_rec)

        return lines


def process_for_training(raw_data, y_key, exclude=[]):
    """
    Returns a dictionary with "X" and "y", two numpy arrays. All data is turned into numeric. Also returns a "categories" field with the list of all features in the order of X.

    y_key is the key of the "y" value - the label. If this value is passed as None (to ready only prediction data) then the y column returned is full of Nones

    exclude is a list of keys to exclude from the raw data.

    raw_data is a list of dictionaries returned by ChunkDictReader.
    """
    # add y_key to exclude
    exclude.append(y_key)

    categories_raw = raw_data[0].keys()
    categories = []
    for cat in categories_raw:
        if cat not in exclude:
            categories.append(cat)

    # sorting keys in order to make sure that they are always the same in the output numpy array
    categories = sorted(categories)
    # for mapping non-numerical values to numeric ones, using the index in their corresponding lists.
    valuemap = {category: [] for category in categories}
    valuemap[y_key] = []

    # the "+ 1" accounts for the missing y_label from the categories
    raw_arr = np.ndarray( (0, len(categories) + 1) )

    # loop through each example in data
    for item in raw_data:
        row = np.zeros( (1, len(categories) + 1) )
        # flag for missing values
        missing = False
        # loop through categories and get their index to insert into row
        for idx, category in enumerate(categories):
            val = item[category]
            # try to convert to float. If it doesn't work, enumerate using valuemap
            try:
                val = float(val)
            except ValueError:
                if val not in valuemap[category]:
                    valuemap[category].append(val)
                val = valuemap[category].index(val)
            except TypeError:
                print "Error: Value for " + category + " is invalid type:", type(val)
                print "Skipping..."
                missing = True

            row[0, idx] = val

        # skip if any missing values
        if missing: continue

        # if y_key is none, it means we are preparing data that isn't for training. Therefore, the y row will be filled with "Nones"
        if y_key is None:
            row[0, len(categories)] = y_key
        else:
            # len(categories) is the last index in row
            row[0, len(categories)] = float(item[y_key])

        # add row to raw_arr numpy array
        raw_arr = np.append(raw_arr, row, axis=0)

    # randomize the order of the array BEFORE separating into y and X
    np.random.shuffle(raw_arr)
    
    # not reshaping y into a (?, 1) matrix because it is not needed
    return {"X": raw_arr[:, 0:-1], "y": raw_arr[:, -1], "categories": categories}
