from datetime import datetime
import time


def extract_date_from_filepath(filepath):
    return filepath.split('/')[-1].split('_')[-1][:-5]


def time_to_int(timestamp):
    timestamp = str(timestamp)
    """Change time stamp to integer in ms."""
    if len(str(timestamp)) < len("2016-08-01 9:30:00.000"):
        timestamp += ".000"
    date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    current_time = int(time.mktime(date.timetuple())*1e3 + date.microsecond/1e3)
    return current_time

