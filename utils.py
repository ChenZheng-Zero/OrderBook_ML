from datetime import datetime
import time


def time_to_int(timestamp):
    """Change time stamp to integer in ms."""
    date = datetime.strptime(str(timestamp), '%Y/%m/%d %H:%M:%S.%f')
    current_time = int(time.mktime(date.timetuple())*1e3 + date.microsecond/1e3)
    return current_time
