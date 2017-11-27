from datetime import datetime
from math import isnan
import numpy as np
import os
import pandas as pd
import time


def extract_features_from_order_books(limit_order_filename, transaction_order_filename, feature_filename,
                                      n_level=10, delta_t=50, delta_T=1000):
    if not os.path.isfile(feature_filename):
        basic_set, timestamps, basic_labels = extract_basic_features_by_d(limit_order_filename, n_level)
        time_insensitive_set = extract_time_insensitive_features(basic_set, n_level)
        time_sensitive_set = extract_time_sensitive_features(limit_order_filename,
                                                             transaction_order_filename,
                                                             n_level, delta_t, delta_T)
        timestamps = timestamps[:len(time_sensitive_set)]
        basic_set = basic_set[:len(time_sensitive_set)]
        time_insensitive_set = time_insensitive_set[:len(time_sensitive_set)]
        labels = basic_labels[:len(time_sensitive_set)]
        save_feature_json(feature_filename, timestamps, basic_set, time_insensitive_set,
                          time_sensitive_set, labels)
    df = pd.read_json(feature_filename, orient="records", lines="True")
    timestamps = df["timestamps"].tolist()
    basic_set = df["basic_set"].tolist()
    time_insensitive_set = df["time_insensitive_set"].tolist()
    time_sensitive_set = df["time_sensitive_set"].tolist()
    labels = df["labels"].tolist()
    return np.array(timestamps), np.array(basic_set), np.array(time_insensitive_set), \
        np.array(time_sensitive_set), np.array(labels)


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
        tmp_v1 = []
        for index in range(delimiter_index[i] + 1, delimiter_index[i] + 1 + n_level):
            tmp_v1.append([limit_order_df["ASK_PRICE"][index], limit_order_df["ASK_SIZE"][index],
                           limit_order_df["BID_PRICE"][index], limit_order_df["BID_SIZE"][index]])
        mid_prices.append((tmp_v1[0][0] + tmp_v1[0][2])/2)
        timestamps.append(change_time_form(limit_order_df["Time"][i]))
        basic_set.append([var for v1_i in tmp_v1 for var in v1_i])
    basic_labels = get_mid_price_labels(mid_prices)
    return basic_set, timestamps, basic_labels


def save_feature_json(feature_filename, timestamps, basic_set, time_insensitive_set,
                      time_sensitive_set, labels):
    feature_dict = {"timestamps": timestamps, "basic_set": basic_set,
                    "time_insensitive_set": time_insensitive_set,
                    "time_sensitive_set": time_sensitive_set, "labels": labels}
    df = pd.DataFrame(data=feature_dict, columns=["timestamps", "basic_set",
                                                  "time_insensitive_set", "time_sensitive_set",
                                                  "labels"])
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
    guaranteed_index = check_n_level(limit_order_df, delimiter_index, n_level=10)
    delimiter_index = delimiter_index[guaranteed_index:]
    return delimiter_index


def check_n_level(limit_order_df, delimiter_index, n_level=10):
    guaranteed_index = 0
    for i in range(len(delimiter_index) - 1):
        count = 0
        if delimiter_index[i + 1] - delimiter_index[i] < n_level:
            guaranteed_index = i + 1
            continue
        for index in range(delimiter_index[i] + 1, delimiter_index[i + 1]):
            if limit_order_df["BID_SIZE"][index] * limit_order_df["ASK_SIZE"][index] > 0:
                count += 1
            else:
                if count < n_level:
                    guaranteed_index = i+1
    return guaranteed_index


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


def get_time_insensitive_v2(v1):
    v2 = [[v1_i[0] - v1_i[2], (v1_i[0] + v1_i[2])/2] for v1_i in v1]
    return [var for v2_i in v2 for var in v2_i]


def get_time_insensitive_v3(v1):
    v3 = [[v1[-1][0] - v1[0][0], v1[0][2] - v1[-1][2],
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


def extract_time_sensitive_features(limit_order_filename, transaction_order_filename,
                                    n_level=10, delta_t=50, delta_T=1000):
    v6 = get_time_sensitive_v6(limit_order_filename, n_level, delta_t)

    limit_ask_density_list, limit_bid_density_list = get_limit_density_list(limit_order_filename, n_level)
    transaction_ask_density_list, transaction_bid_density_list, cancelled_ask_density_list, \
        cancelled_bid_density_list \
        = get_transaction_and_cancelled_density_list(limit_order_filename, transaction_order_filename)
    v7 = get_time_sensitive_v7(limit_ask_density_list, limit_bid_density_list,
                               transaction_ask_density_list, transaction_bid_density_list,
                               cancelled_ask_density_list, cancelled_bid_density_list,
                               delta_t)
    v8 = get_time_sensitive_v8(limit_ask_density_list, limit_bid_density_list,
                               transaction_ask_density_list, transaction_bid_density_list,
                               cancelled_ask_density_list, cancelled_bid_density_list,
                               delta_t, delta_T)
    time_sensitive_features = merge_time_sensitive_features(v6, v7, v8)
    return time_sensitive_features


def get_time_sensitive_v6(limit_order_filename, n_level=10, delta_t=50):
    limit_order_df = pd.read_excel(limit_order_filename)
    delimiter_index = get_delimiter_index(limit_order_df)
    v6 = []
    for i in range(len(delimiter_index) - 1 - delta_t):
        tmp_v6 = []
        for level in range(1, n_level+1):
            curr_index = delimiter_index[i] + level
            after_delta_t_index = delimiter_index[i + delta_t] + level
            tmp_v6.append([limit_order_df["ASK_PRICE"][after_delta_t_index] -
                           limit_order_df["ASK_PRICE"][curr_index],
                           limit_order_df["BID_PRICE"][after_delta_t_index] -
                           limit_order_df["BID_PRICE"][curr_index],
                           limit_order_df["ASK_SIZE"][after_delta_t_index] -
                           limit_order_df["ASK_SIZE"][curr_index],
                           limit_order_df["BID_SIZE"][after_delta_t_index] -
                           limit_order_df["BID_SIZE"][curr_index]])
        v6.append([var for v6_i in tmp_v6 for var in v6_i])
    return v6


def get_time_sensitive_v7(limit_ask_density_list, limit_bid_density_list,
                          transaction_ask_density_list, transaction_bid_density_list,
                          cancelled_ask_density_list, cancelled_bid_density_list,
                          delta_t=50):
    v7 = []
    for i in range(len(cancelled_bid_density_list) - delta_t):
        v7.append([limit_ask_density_list[i + delta_t] - limit_ask_density_list[i],
                   limit_bid_density_list[i + delta_t] - limit_bid_density_list[i],
                   transaction_ask_density_list[i + delta_t] - transaction_ask_density_list[i],
                   transaction_bid_density_list[i + delta_t] - transaction_bid_density_list[i],
                   cancelled_ask_density_list[i + delta_t] - cancelled_ask_density_list[i],
                   cancelled_bid_density_list[i + delta_t] - cancelled_bid_density_list[i]])
    return v7


def get_limit_density_list(limit_order_filename, n_level=10):
    limit_order_df = pd.read_excel(limit_order_filename)
    delimiter_index = get_delimiter_index(limit_order_df)
    limit_ask_density_list = []
    limit_bid_density_list = []
    for i in range(len(delimiter_index) - 1):
        limit_ask_density = 0.0
        limit_bid_density = 0.0
        for index in range(delimiter_index[i] + 1, delimiter_index[i] + 1 + n_level):
            limit_ask_density += limit_order_df["ASK_PRICE"][index] * limit_order_df["ASK_SIZE"][index]
            limit_bid_density += limit_order_df["BID_PRICE"][index] * limit_order_df["BID_SIZE"][index]
        limit_ask_density_list.append(limit_ask_density)
        limit_bid_density_list.append(limit_bid_density)
    return limit_ask_density_list, limit_bid_density_list


def get_transaction_and_cancelled_density_list(limit_order_filename, transaction_order_filename):
    limit_order_df = pd.read_excel(limit_order_filename)
    delimiter_index = get_delimiter_index(limit_order_df)
    transaction_dict = get_transaction_dict(transaction_order_filename)
    transaction_ask_density_list = []
    transaction_bid_density_list = []
    cancelled_ask_density_list = []
    cancelled_bid_density_list = []
    prev_ask_orders = []
    prev_bid_orders = []
    for i in range(len(delimiter_index) - 1):
        curr_ask_orders = []
        curr_bid_orders = []
        timestamp = limit_order_df["Time"][delimiter_index[i] + 1]
        for index in range(delimiter_index[i] + 1, delimiter_index[i + 1]):
            if not isnan(limit_order_df["ASK_PRICE"][index]):
                curr_ask_orders.append({"PRICE": limit_order_df["ASK_PRICE"][index],
                                        "SIZE": limit_order_df["ASK_SIZE"][index]})
            if not isnan(limit_order_df["BID_PRICE"][index]):
                curr_bid_orders.append({"PRICE": limit_order_df["BID_PRICE"][index],
                                        "SIZE": limit_order_df["BID_SIZE"][index]})
        disappeared_ask_orders = get_disappeared_orders(prev_ask_orders, curr_ask_orders)
        transaction_ask_density, cancelled_ask_density = get_transaction_and_cancelled_density(timestamp,
                                                            disappeared_ask_orders, transaction_dict)
        transaction_ask_density_list.append(transaction_ask_density)
        cancelled_ask_density_list.append(cancelled_ask_density)
        disappeared_bid_orders = get_disappeared_orders(prev_bid_orders, curr_bid_orders)
        transaction_bid_density, cancelled_bid_density = get_transaction_and_cancelled_density(timestamp,
                                                            disappeared_bid_orders, transaction_dict)
        transaction_bid_density_list.append(transaction_bid_density)
        cancelled_bid_density_list.append(cancelled_bid_density)
        prev_ask_orders = curr_ask_orders
        prev_bid_orders = curr_bid_orders
    return transaction_ask_density_list, transaction_bid_density_list, cancelled_ask_density_list, \
        cancelled_bid_density_list


def get_disappeared_orders(prev_orders, curr_orders):
    disappeared_orders = {"PRICE": [], "SIZE": []}
    for prev_order in prev_orders:
        is_disappeared = True
        for curr_order in curr_orders:
            if prev_order["PRICE"] == curr_order["PRICE"]:
                is_disappeared = False
                if prev_order["SIZE"] <= curr_order["SIZE"]:
                    break
                else:
                    disappeared_orders["PRICE"].append(prev_order["PRICE"])
                    disappeared_orders["SIZE"].append(prev_order["SIZE"] - curr_order["SIZE"])
        if is_disappeared:
            disappeared_orders["PRICE"].append(prev_order["PRICE"])
            disappeared_orders["SIZE"].append(prev_order["SIZE"])
    return disappeared_orders


def get_transaction_and_cancelled_density(timestamp, disappeared_orders, transaction_dict):
    transaction_density = 0.0
    cancelled_density = 0.0
    for i in range(len(disappeared_orders["PRICE"])):
        if timestamp in transaction_dict:
            is_transaction = False
            for j in range(len(transaction_dict[timestamp]["PRICE"])):
                if disappeared_orders["PRICE"][i] == transaction_dict[timestamp]["PRICE"][j]:
                    transaction_density += (transaction_dict[timestamp]["PRICE"][j] *
                                            transaction_dict[timestamp]["SIZE"][j])
                    is_transaction = True
                    break
            if not is_transaction:
                cancelled_density += disappeared_orders["PRICE"][i] * disappeared_orders["SIZE"][i]
        else:
            cancelled_density += disappeared_orders["PRICE"][i] * disappeared_orders["SIZE"][i]
    return transaction_density, cancelled_density


def get_transaction_dict(transaction_order_filename):
    transaction_order_df = pd.read_excel(transaction_order_filename)
    transaction_dict = {}
    for index in range(len(transaction_order_df["Index"])):
        timestamp = transaction_order_df["Time"][index]
        if transaction_order_df["PRICE"][index] > 0:
            if timestamp in transaction_dict:
                transaction_dict[timestamp]["SIZE"].append(transaction_order_df["SIZE"][index])
                transaction_dict[timestamp]["PRICE"].append(transaction_order_df["PRICE"][index])
            else:
                transaction_dict[timestamp] = {"SIZE": [transaction_order_df["SIZE"][index]],
                                               "PRICE": [transaction_order_df["PRICE"][index]]}
    return transaction_dict


def get_time_sensitive_v8(limit_ask_density_list, limit_bid_density_list,
                          transaction_ask_density_list, transaction_bid_density_list,
                          cancelled_ask_density_list, cancelled_bid_density_list,
                          delta_t=50, delta_T=1000):
    v8 = []
    for i in range(len(cancelled_ask_density_list) - delta_T):
        v8.append([compare_delta_T_and_delta_t(limit_ask_density_list, i, delta_t, delta_T),
                   compare_delta_T_and_delta_t(limit_bid_density_list, i, delta_t, delta_T),
                   compare_delta_T_and_delta_t(transaction_ask_density_list, i, delta_t, delta_T),
                   compare_delta_T_and_delta_t(transaction_bid_density_list, i, delta_t, delta_T)])
    return v8


def compare_delta_T_and_delta_t(list, i, delta_t=50, delta_T=1000):
    return int(list[i + delta_t] > list[i + delta_T])


def merge_time_sensitive_features(v6, v7, v8):
    time_sensitive_features = []
    for i in range(len(v8)):
        time_sensitive_feature = v6[i]
        time_sensitive_feature.extend(v7[i])
        time_sensitive_feature.extend(v8[i])
        time_sensitive_features.append(time_sensitive_feature)
    return time_sensitive_features
