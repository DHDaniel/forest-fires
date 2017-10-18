
# Script that enumerates data and puts it in an SQL-lite database. Numerical data makes learning easier.

from backports import csv
import sqlite3
import copy
import io
from itertools import islice


def month_to_num(month):
    """
    Converts month to numerical value.
    """
    converter = {
    'jan' : 0,
    'feb' : 1,
    'mar' : 2,
    'apr' : 3,
    'may' : 4,
    'jun' : 5,
    'jul' : 6,
    'aug' : 7,
    'sep' : 8,
    'oct' : 9,
    'nov' : 10,
    'dec' : 11
    }
    return converter[month]


def day_to_num(day):
    """
    Converts day of week to numerical value.
    """
    converter = {
    'mon' : 0,
    'tue' : 1,
    'wed' : 2,
    'thu' : 3,
    'fri' : 4,
    'sat' : 5,
    'sun' : 6
    }
    return converter[day]


if __name__ == '__main__':

    filename = raw_input("Main data file name: ")
    db_name = raw_input("New DB name: ")

    original_data = csv.DictReader(io.open(filename, "r", encoding="utf-8"))

    # original headers of data
    headers = copy.copy(original_data.fieldnames)

    # making a database file and storing our original data there
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS fires")
    command = "CREATE TABLE fires ("
    for header in headers:
        # all values being used are numbers
        command += header + " REAL,"

    # label that indicates whether a fire occured, or did not.
    command += "label INTEGER"
    command += ")"
    # create table with all the fields.
    cur.execute(command)

    count = 0
    writesize = 100
    while True:
        count += writesize
        print "Writing data (", count, "items written) ..."
        # read :writesize: lines at a time
        lines = []
        for line in islice(original_data, 0, writesize):
            row = []
            # get dictionary into ordered list and convert necessary values + add in the label row.
            for idx, header in enumerate(headers):
                val = line[header]
                if header == "month":
                    val = month_to_num(val)
                if header == "day":
                    val = day_to_num(val)
                row.append(float(val))

            # determine label from area of fire, which is the last value added to the row.
            if row[-1] > 0:
                label = 1
            else:
                label = 0
            row.append(label)

            lines.append(row)

        if len(lines) == 0: break

        # adding extra '?' in insert command because of the extra "label" field NOT IN headers
        insert_command = "INSERT INTO fires VALUES (" + ",".join(["?" for header in headers]) + ", ?)"
        cur.executemany(insert_command, lines)
        conn.commit()

    conn.close()
