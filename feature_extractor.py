from datetime import datetime
import numpy as np
import pandas as pd
import time


def read_limit_order_book(order_book_filename, n_level, feature_filename):
    df = pd.read_excel(order_book_filename, header=0)
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    v5 = []
    mid_prices = []
    timestamps = []

    level_count = 0
    tmp_v1 = []

    for i in range(len(df["Index"])):
        if df["DELIMITER"][i] == "D":
            tmp_v1 = []
            level_count = 0
            continue
        if df["BID_SIZE"][i] * df["ASK_SIZE"][i] > 0 and level_count < n_level:
            tmp_v1.append([df["ASK_PRICE"][i], df["ASK_SIZE"][i],
                           df["BID_PRICE"][i], df["BID_SIZE"][i]])
            level_count += 1
            if level_count == n_level:
                # use the first mid_price as the mid_price of this timestamp
                mid_prices.append((tmp_v1[0][0] + tmp_v1[0][2])/2)
                timestamps.append(change_time_form(df["Time"][i]))
                flattened_v1 = [var for v1_for_a_moment in tmp_v1 for var in v1_for_a_moment]
                v1.append(flattened_v1)
                v2.append(get_time_insensitive_v2(tmp_v1))
                v3.append(get_time_insensitive_v3(tmp_v1))
                v4.append(get_time_insensitive_v4(tmp_v1))
                v5.append(get_time_insensitive_v5(tmp_v1))
    labels = get_mid_price_labels(mid_prices)
    time_insensitive_features = get_time_insensitive_features(v2, v3, v4, v5)
    feature_dict = {"timestamps": timestamps, "basic_set": v1,
                    "time_insensitive_set": time_insensitive_features,
                    "labels": labels}
    df = pd.DataFrame(feature_dict, columns=["timestamps", "basic_set", "time_insensitive_set", "labels"])
    df.to_json(feature_filename, orient='records', lines=True)
    return np.array(timestamps), np.array(v1), np.array(time_insensitive_features), np.array(labels)


def change_time_form(timestamp):
    date = datetime.strptime(str(timestamp), '%Y/%m/%d %H:%M:%S.%f')
    current_time = int(time.mktime(date.timetuple())*1e3 + date.microsecond/1e3)
    return current_time


def get_mid_price_labels(mid_prices):
    gt = [0]  # let the start label be 0
    if len(mid_prices) == 1:
        return gt
    for i in range(1, len(mid_prices)):
        if mid_prices[i] - mid_prices[i - 1] > 0:
            gt.append(1)
        elif mid_prices[i] - mid_prices[i - 1] == 0:
            gt.append(0)
        else:
            gt.append(-1)
    return gt


def get_time_insensitive_features(v2, v3, v4, v5):
    time_insensitive_features = []
    for i in range(len(v2)):
        time_insensitive_feature = v2[i] + v3[i] + v4[i] + v5[i]
        time_insensitive_features.append(time_insensitive_feature)
    return time_insensitive_features


def get_time_insensitive_v2(v1):
    v2 = [[v1_i[0] - v1_i[2], (v1_i[0] + v1_i[2])/2] for v1_i in v1]
    return [var for v2_i in v2 for var in v2_i]


def get_time_insensitive_v3(v1):
    v3 = [[v1[-1][0] - v1[-1][0], v1[0][2] - v1[-1][2],
           abs(v1[i][0] - v1[i - 1][0]), abs(v1[i][2] - v1[i - 1][2])]
          for i in range(len(v1)) if i > 0]
    return [var for v3_i in v3 for var in v3_i]


def get_time_insensitive_v4(v1):
    P_ask = [v1_i[0] for v1_i in v1]
    V_ask = [v1_i[1] for v1_i in v1]
    P_bid = [v1_i[2] for v1_i in v1]
    V_bid = [v1_i[3] for v1_i in v1]
    return [sum(P_ask)/len(P_ask), sum(P_bid)/len(P_bid),
            sum(V_ask)/len(V_ask), sum(V_bid)/len(V_bid)]


def get_time_insensitive_v5(v1):
    P_ask_P_bid = [v1_i[0] - v1_i[2] for v1_i in v1]
    V_ask_V_bid = [v1_i[1] - v1_i[3] for v1_i in v1]
    return [sum(P_ask_P_bid), sum(V_ask_V_bid)]

