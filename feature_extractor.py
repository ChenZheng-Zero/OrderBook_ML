from datetime import datetime
import numpy as np
import os
import pandas as pd
import time


def extract_features_from_order_books(limit_order_filename, feature_filename, n_level, delta_t=50, delta_T=0):
    if not os.path.isfile(feature_filename):
        basic_set, timestamps, basic_labels = extract_basic_features_by_timestamp(limit_order_filename, n_level)
        time_insensitive_set = extract_time_insensitive_features(basic_set, n_level)
        save_feature_json(feature_filename, basic_set, timestamps, basic_labels, time_insensitive_set)
        return np.array(timestamps), np.array(basic_set), np.array(time_insensitive_set), np.array(basic_labels)
    df = pd.read_json(feature_filename, orient="records", lines="True")
    basic_set = df["basic_set"].tolist()
    timestamps = df["timestamps"].tolist()
    basic_labels = df["labels"].tolist()
    time_insensitive_set = df["time_insensitive_set"].tolist()
    return np.array(timestamps), np.array(basic_set), np.array(time_insensitive_set), np.array(basic_labels)


def extract_basic_features_by_timestamp(limit_order_filename, n_level):
    limit_order_df = pd.read_excel(limit_order_filename)
    current_timestamp = ""
    timestamps = []
    mid_prices = []
    basic_set = []
    tmp_v1 = []
    for index in range(len(limit_order_df["Index"])):
        if limit_order_df["Time"][index] != current_timestamp:
            current_timestamp = limit_order_df["Time"][index]
            tmp_v1 = []
        if len(tmp_v1) < n_level and limit_order_df["BID_SIZE"][index] * limit_order_df["ASK_SIZE"][index] > 0:
            tmp_v1.append([limit_order_df["ASK_PRICE"][index], limit_order_df["ASK_SIZE"][index],
                           limit_order_df["BID_PRICE"][index], limit_order_df["BID_SIZE"][index]])
            if len(tmp_v1) == n_level:
                mid_prices.append((tmp_v1[0][0] + tmp_v1[0][2])/2)
                timestamps.append(change_time_form(limit_order_df["Time"][index]))
                basic_set.append([var for v1_i in tmp_v1 for var in v1_i])
    basic_labels = get_mid_price_labels(mid_prices)
    return basic_set, timestamps, basic_labels


def extract_basic_features_by_d(limit_order_filename, n_level):
    limit_order_df = pd.read_excel(limit_order_filename)
    delimiter_index = get_delimiter_index(limit_order_df)
    timestamps = []
    mid_prices = []
    basic_set = []
    for i in range(len(delimiter_index) - 1):
        if delimiter_index[i + 1] - delimiter_index[i] > n_level:
            tmp_v1 = []
            for index in range(delimiter_index[i] + 1, delimiter_index[i + 1]):
                if limit_order_df["BID_SIZE"][index] * limit_order_df["ASK_SIZE"][index] > 0:
                    tmp_v1.append([limit_order_df["ASK_PRICE"][index], limit_order_df["ASK_SIZE"][index],
                                   limit_order_df["BID_PRICE"][index], limit_order_df["BID_SIZE"][index]])
                if len(tmp_v1) == n_level:
                    mid_prices.append((tmp_v1[0][0] + tmp_v1[0][2])/2)
                    timestamps.append(change_time_form(limit_order_df["Time"][i]))
                    basic_set.append([var for v1_i in tmp_v1 for var in v1_i])
                    break
    basic_labels = get_mid_price_labels(mid_prices)
    return basic_set, timestamps, basic_labels


def save_feature_json(feature_filename, basic_set, timestamps, basic_labels, time_insensitive_set):
    feature_dict = {"timestamps": timestamps, "basic_set": basic_set,
                    "time_insensitive_set": time_insensitive_set, "labels": basic_labels}
    df = pd.DataFrame(data=feature_dict, columns=["timestamps", "basic_set", "time_insensitive_set", "labels"])
    df.to_json(path_or_buf=feature_filename, orient="records", lines=True)


def extract_time_insensitive_features(basic_set, n_level):
    time_insensitive_set = []
    for v1 in basic_set:
        v1 = np.array(v1).reshape(n_level, -1)
        v2 = get_time_insensitive_v2(v1)
        v3 = get_time_insensitive_v3(v1)
        v4 = get_time_insensitive_v4(v1)
        v5 = get_time_insensitive_v5(v1)
        time_insensitive_feature = v2 + v3 + v4 + v5
        time_insensitive_set.append(time_insensitive_feature)
    return time_insensitive_set


def get_delimiter_index(limit_order_df):
    delimiter_index = [-1] # assume there is a D before index 0
    for i in range(len(limit_order_df["DELIMITER"])):
        if limit_order_df["DELIMITER"][i] == "D":
            delimiter_index.append(i)
    return delimiter_index


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


def merge_time_insensitive_features(v2, v3, v4, v5):
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
    p_ask = [v1_i[0] for v1_i in v1]
    v_ask = [v1_i[1] for v1_i in v1]
    p_bid = [v1_i[2] for v1_i in v1]
    v_bid = [v1_i[3] for v1_i in v1]
    return [sum(p_ask)/len(p_ask), sum(p_bid)/len(p_bid),
            sum(v_ask)/len(v_ask), sum(v_bid)/len(v_bid)]


def get_time_insensitive_v5(v1):
    p_ask_p_bid = [v1_i[0] - v1_i[2] for v1_i in v1]
    v_ask_v_bid = [v1_i[1] - v1_i[3] for v1_i in v1]
    return [sum(p_ask_p_bid), sum(v_ask_v_bid)]


def get_time_sensitive_v6(limit_order_filename, n_level, delta_t=50):
    limit_order_df = pd.read_excel(limit_order_filename)
    delimiter_index = get_delimiter_index(limit_order_df)
    v6 = []
    for i in range(len(delimiter_index)):
        if i + delta_t < len(delimiter_index) - 1:
            if (delimiter_index[i + 1] - delimiter_index[i] > n_level and
                    delimiter_index[i + delta_t + 1] - delimiter_index[i + delta_t] > n_level):
                tmp_v6 = []
                for level in range(1, n_level+1):
                    curr_index = delimiter_index[i] + level
                    after_delta_t_index = delimiter_index[i + delta_t] + level
                    if (limit_order_df["BID_SIZE"][curr_index] * limit_order_df["ASK_SIZE"][curr_index] > 0 and
                                (limit_order_df["BID_SIZE"][after_delta_t_index] *
                                     limit_order_df["ASK_SIZE"][after_delta_t_index]) > 0):
                        tmp_v6.append([limit_order_df["ASK_PRICE"][after_delta_t_index] -
                                       limit_order_df["ASK_PRICE"][after_delta_t_index],
                                       limit_order_df["BID_PRICE"][after_delta_t_index] -
                                       limit_order_df["BID_PRICE"][after_delta_t_index],
                                       limit_order_df["ASK_SIZE"][after_delta_t_index] -
                                       limit_order_df["ASK_SIZE"][after_delta_t_index],
                                       limit_order_df["BID_SIZE"][after_delta_t_index] -
                                       limit_order_df["BID_SIZE"][after_delta_t_index]])
                if len(tmp_v6) == n_level:
                    v6.append([var for v6_i in tmp_v6 for var in v6_i])
    return v6
