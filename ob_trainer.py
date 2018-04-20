from pre_processor import get_samples_index
import mid_spread_label
import feature_extractor
import execution_strategy
import numpy as np
import svm_trainer
import os
import pandas as pd
import pdb


def train_order_books(input_dir, stock, dates, event_time, label_type,
                      time_interval=100, n_level=10):
    basic_set = []
    time_insensitive_set = []
    time_sensitive_set = []
    labels = []

    for date in dates:
        limit_order_filenames = sorted([os.path.join(input_dir, file) for file in os.listdir(input_dir)
                                        if "_OB_" + date in file])
        print("Limit order books: ", limit_order_filenames)
        limit_order_filename = os.path.join(input_dir, stock + "_OB_" + date + ".xlsx")
        trd_order_filename = os.path.join(input_dir, stock + "_TRD_" + date + ".xlsx")
        cancel_order_filename = os.path.join(input_dir, stock + "_Order_Cancel_" + date + ".xlsx")
        submission_filename = os.path.join(input_dir, stock + "_SUB_" + date + ".xlsx")
        feature_filename = "../output_zheng/" + stock + "_" + date + "_" + event_time + ".json"

        if not os.path.isfile(feature_filename):
            extractor = feature_extractor.FeatureExtractor(
                limit_order_filenames=limit_order_filenames,
                trd_order_filename=trd_order_filename,
                cancel_order_filename=cancel_order_filename,
                submission_filename=submission_filename,
                feature_filename=feature_filename, event_time=event_time,
                time_interval=time_interval, n_level=n_level)
            extractor.extract_features()
        threshold = 0.75
        midspread_filename = feature_filename[:-5] + "_midspread_" + str(threshold) + ".json"

        if not os.path.isfile(midspread_filename):
            mid_spread_label.get_mid_spread_labels(feature_filename, midspread_filename, threshold=threshold)

        # read features
        df = pd.read_json(midspread_filename, orient="records", lines="True")
        ob_basic_set = np.array(df["basic_set"].tolist())
        ob_time_insensitive_set = np.array(df["time_insensitive_set"].tolist())
        ob_time_sensitive_set = np.array(df["time_sensitive_set"].tolist())
        if label_type == "mid":
            ob_labels = np.array(df["mid_price_labels"].tolist())
        elif label_type == "spread":
            ob_labels = np.array(df["spread_crossing_labels"].tolist())
        else:
            ob_labels = np.array(df["mid_spread_labels"].tolist())

        print("Order book {} has {} data points".format(limit_order_filename.split('/')[-1], len(ob_labels)))
        basic_set.extend(ob_basic_set)
        time_insensitive_set.extend(ob_time_insensitive_set)
        time_sensitive_set.extend(ob_time_sensitive_set)
        labels.extend(ob_labels)

    basic_set = np.array(basic_set)
    time_insensitive_set = np.array(time_insensitive_set)
    time_sensitive_set = np.array(time_sensitive_set)
    labels = np.array(labels)

    train_index, test_index, idx = get_samples_index(labels, split=0.25)
    features = np.concatenate((basic_set, time_insensitive_set, time_sensitive_set), axis=1)
    selected_train_data = features[train_index]
    selected_train_labels = labels[train_index]
    selected_test_data = features[test_index]
    selected_test_labels = labels[test_index]

    # max_info_indices = feature_selection(selected_data, selected_labels)
    # selected_data = selected_data[:, max_info_indices]
    C = [1, 1e1, 1e2, 1e3, 1e4, 1e5]
    G = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    # optimal 5e4, 1e-7
    # optimal 10000, 1e-8, accuracy 78%, precision 71%
    for c in C:
        for g in G:
            print("SVM c = {}".format(c) + " g = {}".format(g))
            score, model = svm_trainer.train_svm(train_data=selected_train_data, train_labels=selected_train_labels, \
                                                 test_data=selected_test_data, test_labels=selected_test_labels, c=c, kernel='rbf', g=g)

    # execution strategy based on ground truth
    test_data = features[idx:]
    test_labels = labels[idx:]
    cash = execution_strategy.execution(full_test_data=test_data, full_test_labels=test_labels,
                                        max_holdings=200, unit=1, tick_increment=0.01)

# def train_order_book(limit_order_filename, trd_order_filename, cancel_order_filename,
#                      submission_filename, feature_filename, event_time, label_type,
#                      n_level=10, time_interval=100):
#     extractor = feature_extractor.FeatureExtractor(
#         limit_order_filename=limit_order_filename,
#         trd_order_filename=trd_order_filename,
#         cancel_order_filename=cancel_order_filename,
#         submission_filename=submission_filename,
#         feature_filename=feature_filename, event_time=event_time,
#         time_interval=time_interval, n_level=n_level)
#     timestamps, basic_set, time_insensitive_set, time_sensitive_set, \
#         mid_price_labels, spread_crossing_labels = extractor.extract_features()
#     print("Order book {} has {} data points".format(limit_order_filename.split('/')[-1], len(mid_price_labels)))
#
#     if label_type == 'mid':
#         labels = mid_price_labels
#     else:
#         labels = spread_crossing_labels
#
#     train_index, test_index, idx = get_samples_index(labels, split=0.25)
#     features = np.concatenate((basic_set, time_insensitive_set, time_sensitive_set), axis=1)
#     selected_train_data = features[train_index]
#     selected_train_labels = labels[train_index]
#     selected_test_data = features[test_index]
#     selected_test_labels = labels[test_index]
#
#     # max_info_indices = feature_selection(selected_data, selected_labels)
#     # selected_data = selected_data[:, max_info_indices]
#     C = [1, 1e1, 1e2, 1e3, 1e4, 1e5]
#     G = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
#     # optimal 5e4, 1e-7
#     # optimal 10000, 1e-8, accuracy 78%, precision 71%
#     for c in C:
#         for g in G:
#             print("SVM c = {}".format(c) + " g = {}".format(g))
#             score, model = svm_trainer.train_svm(train_data=selected_train_data, train_labels=selected_train_labels, \
#                 test_data=selected_test_data, test_labels=selected_test_labels, c=c, kernel='rbf', g=g)
#
#     # execution strategy based on ground truth
#     test_data = features[idx:]
#     test_labels = labels[idx:]
#     cash = execution_strategy.execution(full_test_data=test_data, full_test_labels=test_labels,
#                                         max_holdings=200, unit=1, tick_increment=0.01)

