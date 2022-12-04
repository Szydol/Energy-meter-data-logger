import os
import sqlite3
import time


def data_logger(today_date, presence):
    try:
        conn = sqlite3.connect('measurement.db')
        curs = conn.cursor()
        print("Connected to SQLite")
        sqlite_insert_with_param = """INSERT INTO user_detection
                          (date, presence) 
                          VALUES (?,?);"""
        data_tuple = (today_date, presence)
        curs.execute(sqlite_insert_with_param, data_tuple)
        conn.commit()
        print("Variables inserted successfully")
        curs.close()
    except sqlite3.Error as error:
        print("Failed to insert Python variable into sqlite table", error)
    finally:
        if conn:
            conn.close()
            print("The SQLite connection is closed")


def main():
    timestamp = time.strftime('%H:%M %d-%m-%Y')
    hostname = "192.168.8.121"
    response = os.system("ping -c 1 " + hostname)

    if response == 0:
        print(hostname, 'is up!')
        user_home = True
    else:
        print(hostname, 'is down!')
        user_home = False

    data_logger(timestamp, user_home)


if __name__ == "__main__":
    main()
