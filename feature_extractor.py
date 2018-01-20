from datetime import datetime
from math import isnan
import numpy as np
import os
import pandas as pd
import time


min_time_interval = 100


def extract_features(limit_order_filename, transaction_order_filename,
                     feature_filename,  time_interval=100, n_level=10,
                     delta_t=50, delta_T=1000):
    if not os.path.isfile(feature_filename):
        limit_order_df = pd.read_excel(limit_order_filename)
        delimiter_indices = get_delimiter_indices(limit_order_df)
        guaranteed_index = check_n_level(limit_order_df, delimiter_indices, n_level)
        duplicate_timestamps = get_all_timestamps_in_int(limit_order_df)
        time_interval_start_indices = get_time_interval_start_indices(duplicate_timestamps,
                                                                      guaranteed_index)
        unit_time_features, unit_timestamps, unit_mid_prices = extract_unit_time_set(limit_order_df,
                                                                                     time_interval_start_indices,
                                                                                     duplicate_timestamps, n_level)
        basic_set, timestamps, basic_labels = extract_basic_set(unit_time_features, unit_timestamps,
                                                                unit_mid_prices, time_interval)
        time_insensitive_set = extract_time_insensitive_features(basic_set, n_level)
        time_sensitive_set = extract_time_sensitive_set(limit_order_df, transaction_order_filename,
                                                        time_interval_start_indices,
                                                        unit_time_features, time_interval,
                                                        delta_t, delta_T, n_level)
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


def extract_unit_time_set(limit_order_df, time_interval_start_indices,
                          duplicate_timestamps, n_level):
    """Extract unit time features from the limit order book."""
    assert(len(time_interval_start_indices) > 0)
    unit_timestamps = []
    unit_time_features = []
    unit_mid_prices = []
    for i in time_interval_start_indices:
        unit_timestamps.append(duplicate_timestamps[i])
        unit_time_feature = []
        for index in range(i, i + n_level):
            unit_time_feature.append([limit_order_df["ASK_PRICE"][index],
                                      limit_order_df["ASK_SIZE"][index],
                                      limit_order_df["BID_PRICE"][index],
                                      limit_order_df["BID_SIZE"][index]])
        unit_mid_prices.append((unit_time_feature[0][0] + unit_time_feature[0][2])/2)
        unit_time_features.append(np.array(unit_time_feature).reshape(1, 40).tolist()[0])
    return unit_time_features, unit_timestamps, unit_mid_prices


def extract_basic_set(unit_time_features, unit_timestamps, unit_mid_prices, time_interval):
    """Extract basic set from unit_time_features according to the given time_interval."""
    stop = time_interval/min_time_interval
    basic_labels = get_mid_price_labels(unit_mid_prices[::stop])
    return unit_time_features[::stop], unit_timestamps[::stop], basic_labels


def extract_time_insensitive_features(basic_set, n_level):
    """Extract time insensitive features."""
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


def extract_time_sensitive_set(limit_order_df, transaction_order_filename,
                               time_interval_start_indices, unit_time_features, time_interval,
                               delta_t=50, delta_T=1000, n_level=10):
    """Get time sensitive set."""
    v6 = get_time_sensitive_v6(unit_time_features, delta_t)

    limit_ask_density_list, limit_bid_density_list = get_limit_density_list(limit_order_df,
                                                                            time_interval_start_indices,
                                                                            n_level)
    transaction_ask_density_list, transaction_bid_density_list, \
    cancelled_ask_density_list, cancelled_bid_density_list \
        = get_transaction_and_cancelled_density_list(limit_order_df,
                                                     transaction_order_filename)
    stop = time_interval / min_time_interval
    # pick density every time interval rather than every delimiter
    limit_ask_density_list = np.array(limit_ask_density_list)[time_interval_start_indices].tolist()[::stop]
    limit_bid_density_list = np.array(limit_bid_density_list)[time_interval_start_indices].tolist()[::stop]
    transaction_ask_density_list = np.array(transaction_ask_density_list)[time_interval_start_indices].tolist()[::stop]
    transaction_bid_density_list = np.array(transaction_bid_density_list)[time_interval_start_indices].tolist()[::stop]
    cancelled_ask_density_list = np.array(cancelled_ask_density_list)[time_interval_start_indices].tolist()[::stop]
    cancelled_bid_density_list = np.array(cancelled_bid_density_list)[time_interval_start_indices].tolist()[::stop]

    v7 = get_time_sensitive_v7(limit_ask_density_list, limit_bid_density_list,
                               transaction_ask_density_list, transaction_bid_density_list,
                               cancelled_ask_density_list, cancelled_bid_density_list,
                               delta_t)
    v8 = get_time_sensitive_v8(limit_ask_density_list, limit_bid_density_list,
                               transaction_ask_density_list, transaction_bid_density_list,
                               delta_t, delta_T)
    time_sensitive_features = merge_time_sensitive_features(v6, v7, v8)
    return time_sensitive_features


def get_time_interval_start_indices(duplicate_timestamps, guaranteed_index):
    """Get indices of all start states in every time interval."""
    next_time = min_time_interval
    time_interval_start_indices = []
    for i in range(len(duplicate_timestamps)):
        if duplicate_timestamps[i] >= next_time and i >= guaranteed_index:
            time_interval_start_indices.append(i)
            next_time += min_time_interval
    return time_interval_start_indices


def get_mid_price_labels(mid_prices):
    """Get the labels"""
    gt = [0]  # let the start label be 0
    if len(mid_prices) == 1:
        return gt
    for i in range(1, len(mid_prices)):
        if mid_prices[i] - mid_prices[i - 1] > 0:
            gt.append(1)
        else:
            gt.append(0)
    return gt


def get_time_insensitive_v2(v1):
    """Get v2 from v1."""
    v2 = [[v1_i[0] - v1_i[2], (v1_i[0] + v1_i[2])/2] for v1_i in v1]
    return [var for v2_i in v2 for var in v2_i]


def get_time_insensitive_v3(v1):
    """Get v3 from v1."""
    v3 = [[v1[-1][0] - v1[0][0], v1[0][2] - v1[-1][2],
           abs(v1[i][0] - v1[i - 1][0]), abs(v1[i][2] - v1[i - 1][2])]
          for i in range(len(v1)) if i > 0]
    return [var for v3_i in v3 for var in v3_i]


def get_time_insensitive_v4(v1):
    """Get v4 from v1."""
    p_ask = [v1_i[0] for v1_i in v1]
    v_ask = [v1_i[1] for v1_i in v1]
    p_bid = [v1_i[2] for v1_i in v1]
    v_bid = [v1_i[3] for v1_i in v1]
    return [sum(p_ask)/len(p_ask), sum(p_bid)/len(p_bid),
            sum(v_ask)/len(v_ask), sum(v_bid)/len(v_bid)]


def get_time_insensitive_v5(v1):
    """Get v5 from v1."""
    p_ask_p_bid = [v1_i[0] - v1_i[2] for v1_i in v1]
    v_ask_v_bid = [v1_i[1] - v1_i[3] for v1_i in v1]
    return [sum(p_ask_p_bid), sum(v_ask_v_bid)]


def get_time_sensitive_v6(unit_time_features, delta_t=50):
    """Get v6 from unit time features."""
    v6 = []
    for i in range(len(unit_time_features) - delta_t):
        derivative = np.array(unit_time_features[i + delta_t]) - \
                     np.array(unit_time_features[i])
        v6.append(derivative.tolist())
    return v6


def get_time_sensitive_v7(limit_ask_density_list, limit_bid_density_list,
                          transaction_ask_density_list, transaction_bid_density_list,
                          cancelled_ask_density_list, cancelled_bid_density_list,
                          delta_t=50):
    """Get v7."""
    v7 = []
    for i in range(len(cancelled_bid_density_list) - delta_t):
        v7.append([limit_ask_density_list[i + delta_t] - limit_ask_density_list[i],
                   limit_bid_density_list[i + delta_t] - limit_bid_density_list[i],
                   transaction_ask_density_list[i + delta_t] - transaction_ask_density_list[i],
                   transaction_bid_density_list[i + delta_t] - transaction_bid_density_list[i],
                   cancelled_ask_density_list[i + delta_t] - cancelled_ask_density_list[i],
                   cancelled_bid_density_list[i + delta_t] - cancelled_bid_density_list[i]])
    return v7


def get_time_sensitive_v8(limit_ask_density_list, limit_bid_density_list,
                          transaction_ask_density_list, transaction_bid_density_list,
                          delta_t=50, delta_T=1000):
    """Get v8."""
    v8 = []
    for i in range(len(transaction_ask_density_list) - delta_T):
        v8.append([compare_delta_T_and_delta_t(limit_ask_density_list, i, delta_t, delta_T),
                   compare_delta_T_and_delta_t(limit_bid_density_list, i, delta_t, delta_T),
                   compare_delta_T_and_delta_t(transaction_ask_density_list, i, delta_t, delta_T),
                   compare_delta_T_and_delta_t(transaction_bid_density_list, i, delta_t, delta_T)])
    return v8


def get_limit_density_list(limit_order_df, time_interval_start_indices, n_level=10):
    """Get bid/ask density."""
    limit_ask_density_list = []
    limit_bid_density_list = []
    for i in range(len(time_interval_start_indices)):
        limit_ask_density = 0.0
        limit_bid_density = 0.0
        for index in range(time_interval_start_indices[i], time_interval_start_indices[i] + n_level):
            limit_ask_density += limit_order_df["ASK_PRICE"][index] * limit_order_df["ASK_SIZE"][index]
            limit_bid_density += limit_order_df["BID_PRICE"][index] * limit_order_df["BID_SIZE"][index]
        limit_ask_density_list.append(limit_ask_density)
        limit_bid_density_list.append(limit_bid_density)
    return limit_ask_density_list, limit_bid_density_list


def get_transaction_and_cancelled_density_list(limit_order_df,
                                               transaction_order_filename):
    """Get transaction and cancelled density for every delimiter."""
    delimiter_index = get_delimiter_indices(limit_order_df)
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
                                                                                               disappeared_ask_orders,
                                                                                               transaction_dict)
        transaction_ask_density_list.append(transaction_ask_density)
        cancelled_ask_density_list.append(cancelled_ask_density)
        disappeared_bid_orders = get_disappeared_orders(prev_bid_orders, curr_bid_orders)
        transaction_bid_density, cancelled_bid_density = get_transaction_and_cancelled_density(timestamp,
                                                                                               disappeared_bid_orders,
                                                                                               transaction_dict)
        transaction_bid_density_list.append(transaction_bid_density)
        cancelled_bid_density_list.append(cancelled_bid_density)
        prev_ask_orders = curr_ask_orders
        prev_bid_orders = curr_bid_orders
    return transaction_ask_density_list, transaction_bid_density_list, \
        cancelled_ask_density_list, cancelled_bid_density_list


def get_disappeared_orders(prev_orders, curr_orders):
    """Find disappeared orders in the last delimiter."""
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
    """Get transaction and cancelled density for last delimiter."""
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
    """Get a transaction dictionary to calculate transaction density later."""
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


def time_to_int(timestamp):
    """Change time stamp to integer in ms."""
    date = datetime.strptime(str(timestamp), '%Y/%m/%d %H:%M:%S.%f')
    current_time = int(time.mktime(date.timetuple())*1e3 + date.microsecond/1e3)
    return current_time


def get_delimiter_indices(limit_order_df):
    """Get all D's indices in the limit order book"""
    delimiter_indices = [-1] # assume there is a D before index 0
    for i in range(len(limit_order_df["DELIMITER"])):
        if limit_order_df["DELIMITER"][i] == "D":
            delimiter_indices.append(i)
    return delimiter_indices[:len(delimiter_indices) - 1]


def get_all_timestamps_in_int(limit_order_df):
    """Change all timestamps to integers."""
    duplicate_timestamps = []
    for i in range(len(limit_order_df["Time"])):
        duplicate_timestamps.append(time_to_int(limit_order_df["Time"][i]))
    return duplicate_timestamps


def compare_delta_T_and_delta_t(list, i, delta_t=50, delta_T=1000):
    """compare delta_T and delta_t."""
    return int(list[i + delta_t] > list[i + delta_T])


def merge_time_sensitive_features(v6, v7, v8):
    time_sensitive_features = []
    for i in range(len(v8)):
        time_sensitive_feature = v6[i]
        time_sensitive_feature.extend(v7[i])
        time_sensitive_feature.extend(v8[i])
        time_sensitive_features.append(time_sensitive_feature)
    return time_sensitive_features


def check_n_level(limit_order_df, delimiter_indices, n_level=10):
    """Find the first index that satisfy the desired level."""
    guaranteed_index = 0
    for i in range(len(delimiter_indices) - 1):
        count = 0
        if delimiter_indices[i + 1] - delimiter_indices[i] < n_level:
            guaranteed_index = delimiter_indices[i + 1]
            continue
        for index in range(delimiter_indices[i] + 1, delimiter_indices[i + 1]):
            if limit_order_df["BID_SIZE"][index] * limit_order_df["ASK_SIZE"][index] > 0:
                count += 1
            else:
                if count < n_level:
                    guaranteed_index = delimiter_indices[i + 1]
    return guaranteed_index


def save_feature_json(feature_filename, timestamps, basic_set, time_insensitive_set,
                      time_sensitive_set, labels):
    """Save the json."""
    feature_dict = {"timestamps": timestamps, "basic_set": basic_set,
                    "time_insensitive_set": time_insensitive_set,
                    "time_sensitive_set": time_sensitive_set, "labels": labels}
    df = pd.DataFrame(data=feature_dict, columns=["timestamps", "basic_set",
                                                  "time_insensitive_set", "time_sensitive_set",
                                                  "labels"])
    df.to_json(path_or_buf=feature_filename, orient="records", lines=True)
