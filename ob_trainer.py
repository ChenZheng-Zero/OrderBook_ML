from pre_processor import get_samples_index
from sklearn.model_selection import train_test_split
import feature_extractor
import execution_strategy
import mm_strategy
import numpy as np
import svm_trainer
import utils
import datetime
import time
import bisect
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import os
import pandas as pd
import pdb


def extract_features_from_order_books(input_dir, stock, dates, span_type, event_time, label_type,
                      time_interval=100, n_level=10):
    timestamps = []
    basic_set = []
    time_insensitive_set = []
    time_sensitive_set = []
    labels = []

    limit_order_filenames = []
    for date in dates:
        limit_order_filename = os.path.join(input_dir, stock + "_OB_" + date + ".xlsx")
        limit_order_filenames.append(limit_order_filename)

    trd_order_filename = os.path.join(input_dir, stock + "_TRD_" + dates[0][:-1] + ".xlsx")
    cancel_order_filename = os.path.join(input_dir, stock + "_Cancel_" + dates[0][:-1] + ".xlsx")
    submission_filename = os.path.join(input_dir, stock + "_SUB_" + dates[0][:-1] + ".xlsx")

    feature_filename = "../output/" + stock + "_" + dates[0][:-1] + span_type + "_" + event_time + ".json"

    if not os.path.isfile(feature_filename):
        extractor = feature_extractor.FeatureExtractor(
            limit_order_filenames=limit_order_filenames,
            trd_filename=trd_order_filename,
            cancel_order_filename=cancel_order_filename,
            submission_filename=submission_filename,
            feature_filename=feature_filename, 
            event_time=event_time,
            time_interval=time_interval, 
            n_level=n_level)
        extractor.extract_features()

    #threshold = 0.75
    #midspread_filename = feature_filename[:-5] + "_midspread_" + str(threshold) + ".json"

    #if not os.path.isfile(midspread_filename):
    #    mid_spread_label.get_mid_spread_labels(feature_filename, midspread_filename, threshold=threshold)

    # read features
    df = pd.read_json(feature_filename, orient="records", lines="True", convert_dates = False)
    ob_timestamps = np.array(df["timestamps"].tolist())
    ob_basic_set = np.array(df["basic_set"].tolist())
    ob_time_insensitive_set = np.array(df["time_insensitive_set"].tolist())
    ob_time_sensitive_set = np.transpose([0]*ob_basic_set.shape[0])

    # read labels
    if float(label_type) == 0:
        ob_labels = np.array(df["mid_price_labels"].tolist())[:, 0]
    elif float(label_type) == 0.25:
        ob_labels = np.array(df["mid_price_labels"].tolist())[:, 1]
    elif float(label_type) == 0.5:
        ob_labels = np.array(df["mid_price_labels"].tolist())[:, 2]
    elif float(label_type) == 0.75:
        ob_labels = np.array(df["mid_price_labels"].tolist())[:, 3]
    else:
        ob_labels = np.array(df["mid_price_labels"].tolist())[:, 4]

    print("Order book {} has {} data points".format(limit_order_filename.split('/')[-1], len(ob_labels)))
    timestamps.extend(ob_timestamps)
    basic_set.extend(ob_basic_set)
    time_insensitive_set.extend(ob_time_insensitive_set)
    time_sensitive_set.extend(ob_time_sensitive_set)
    labels.extend(ob_labels)

    return np.array(timestamps), np.array(basic_set), np.array(time_insensitive_set), np.array(time_sensitive_set), np.array(labels)
    

def train_model_test_strategy(timestamps, basic_set, time_insensitive_set, time_sensitive_set, feature_level, labels, dates):
    if int(feature_level) == 1:
        features = basic_set
    elif int(feature_level) == 2:
        features = np.concatenate((basic_set, time_insensitive_set), axis=1)
    elif int(feature_level) == 3:
        features = np.concatenate((basic_set, time_insensitive_set, time_sensitive_set), axis=1)

    # get the initial time
    init_time = datetime.datetime(utils.int_to_time(timestamps[0]).year, utils.int_to_time(timestamps[0]).month, \
        utils.int_to_time(timestamps[0]).day, 9, 30, 0, 0)
    init_time_int = int(time.mktime(init_time.timetuple())*1e3 + init_time.microsecond/1e3)

    train_interval = 20
    test_interval = 5
    right = bisect.bisect_right(timestamps, init_time_int+(train_interval+test_interval)*60*1000 , lo=0, hi=len(timestamps))-1
    mid = bisect.bisect_right(timestamps, init_time_int+train_interval*60*1000, lo=0, hi=right)-1
    left = bisect.bisect_left(timestamps, init_time_int, lo=0, hi=mid)
    if timestamps[left] != init_time_int:
        left = left + 1

    while init_time_int+(train_interval+test_interval)*60*1000 <= timestamps[-1]:
        temp_labels = labels[left:mid]
        # train_index, val_index, idx = get_samples_index(temp_labels, split=0.25)
        # selected_train_data = features[train_index]
        # selected_train_labels = labels[train_index]
        # selected_val_data = features[val_index]
        # selected_val_labels = labels[val_index]
        
        selected_train_data, selected_val_data, selected_train_labels, selected_val_labels = \
            train_test_split(features[left:mid], temp_labels, test_size=0.25, shuffle=False)
        
        print("start training...")
        # max_info_indices = feature_selection(selected_data, selected_labels)
        # selected_data = selected_data[:, max_info_indices]
        C = [1e3]
        G = [1e-5, 1e-6, 1e-7, 1e-8]
        acc = 0
        res_model = []
        for c in C:
            for g in G:
                print("SVM c = {}".format(c) + " g = {}".format(g))
                score, model = svm_trainer.train_svm(train_data=selected_train_data, train_labels=selected_train_labels, \
                                                 test_data=selected_val_data, test_labels=selected_val_labels, c=c, kernel='rbf', g=g)
                if score > acc:
                    acc = score
                    res_model = model
                print(score)
        print("saving model...")

        # execution strategy based on ground truth
        print("testing...")
        test_data = features[mid+1:right]
        test_timestamp = timestamps[mid+1:right]
        test_gt_labels = labels[mid+1:right]

        test_pred_labels = res_model.predict(test_data)
        accuracy = accuracy_score(test_gt_labels, test_pred_labels)
        precision = precision_score(test_gt_labels, test_pred_labels, average=None)
        cnf_matrix = confusion_matrix(test_gt_labels, test_pred_labels)
        print("SVM accuracy is {}".format(accuracy))
        print("SVM precision is {}".format(precision))
        print("confusion matrix: ")
        print(cnf_matrix)

        print("testing strategy...")
        trd_filename = os.path.join("../GOOG_0817/", "GOOG" + "_TRD_" + dates[0][:-1] + ".xlsx")
        print("running the baseline mm strategy...")
        baseline_cash = mm_strategy.market_making(trd_filename = trd_filename, full_test_data=test_data, full_test_timestamp=test_timestamp, \
            full_test_labels=test_gt_labels, smart=False, max_holdings=200, unit=1, spread=0.01)
        print("running the smart mm strategy based on prediction...")
        pred_cash = mm_strategy.market_making(trd_filename = trd_filename, full_test_data=test_data, full_test_timestamp=test_timestamp, \
            full_test_labels=test_pred_labels, smart=True, max_holdings=200, unit=1, spread=0.01)
        print("running the smart mm strategy based on ground truth...")
        gt_cash = mm_strategy.market_making(trd_filename = trd_filename, full_test_data=test_data, full_test_timestamp=test_timestamp, \
            full_test_labels=test_gt_labels, smart=True, max_holdings=200, unit=1, spread=0.01)  

        # pdb.set_trace()
        # update for next interval
        init_time_int = init_time_int + test_interval*60*1000
        left = bisect.bisect_left(timestamps, init_time_int, lo=left, hi=mid)
        if timestamps[left] != init_time_int:
            left = left + 1
        mid = right
        right = bisect.bisect_right(timestamps, init_time_int+(train_interval+test_interval)*60*1000 , lo=mid, hi=len(timestamps))-1


