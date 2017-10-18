# Program that adds additional data to the forest fires database.

import sqlite3

def month_to_season(monthnum):
    """
    Converts month numerical value to season numerical value.
    0 - Winter
    1 - Spring
    2 - Summer
    3 - Fall
    """
    if (0 <= monthnum <= 1) or (monthnum == 11):
        return 0
    elif 2 <= monthnum <= 4:
        return 1
    elif 5 <= monthnum <= 7:
        return 2
    elif 8 <= monthnum <= 10:
        return 3


def update_item(item, fieldnames, new_fieldnames):
    """
    Takes in a single database record :item: and processes all the data that it must process, generating a new list (ordered in new_fieldnames order) of values for updating.
    """
    # make sure item is mutable
    to_update = [None for field in new_fieldnames]

    # add a season value to the season column
    monthnum = item[fieldnames.index("month")]
    season = month_to_season(monthnum)
    to_update[new_fieldnames.index("season")] = season


    # place rowid as last element, needed when updating the values in database
    rowid = item[0]
    to_update.append(rowid)

    return tuple(to_update)

if __name__ == '__main__':

    db_name = raw_input("Database filename: ")
    table_name = raw_input("Database table name: ")

    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    # getting the ordered fieldnames in the database for reference when retrieving them
    fieldnames = ["rowid"]
    fields = cur.execute("PRAGMA table_info(" + table_name + ")")
    for field in fields:
        # name is in the second position of the tuple
        fieldnames.append(field[1])


    # new fields to add to the database. Can't use executemany because we are not explicitly setting a value.
    new_columns = ["season"]

    for col in new_columns:
        try:
            cur.execute("ALTER TABLE " + table_name + " ADD COLUMN " + col)
        except sqlite3.OperationalError:
            print "Column", col, "already exists. Skipping creation..."
            continue


    # adding new columns to fieldnames
    fieldnames += new_columns


    # forming the update command that we will use to set the values for the new columns that have been created.
    update_command = "UPDATE " + table_name + " SET "
    for col in new_columns:
        update_command += col + "= ?,"
    # remove trailing comma
    update_command = update_command[:-1]
    update_command += " WHERE rowid = ?"

    # step is the number of items to read at the same time. Only necessary if working with a really large dataset
    counter = 0
    step = 2000
    while True:
        items = cur.execute("SELECT rowid, * FROM " + table_name + " ORDER BY rowid LIMIT ? OFFSET ?", (step, counter))
        # counter to see how many items were actually retrieved
        new_items = []
        temp_counter = 0
        for item in items:
            temp_counter += 1
            # generate the list of things to update, and append it to the list of updates (which will be updated via executemany() )
            to_update = update_item(item, fieldnames, new_columns)
            new_items.append(to_update)

        # update items and write to database
        cur.executemany(update_command, new_items)
        conn.commit()

        # if less items were retrieved than the step count, this means that we have reached the end and must end the loop
        if temp_counter < step: break

        counter += step

    print "Database successfully updated."
    conn.close()
