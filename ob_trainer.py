from pre_processor import get_samples_index
from pre_processor import feature_selection
import feature_extractor
import execution_strategy
import numpy as np
import svm_trainer
import os
import utils
import pdb


def train_all_days(input_folder, output_folder, n_level=10):
    limit_order_books = [file for file in os.listdir(input_folder)
                         if file.startswith("GOOGL_OB")]
    print("Limit order books: ", limit_order_books)

    basic_set = []
    time_insensitive_set = []
    labels = []

    for limit_order_book in limit_order_books:
        date = utils.extract_date_from_filepath(limit_order_book)
        feature_filepath = output_folder + date + "_" + str(n_level) + ".json"
        limit_order_filename = input_folder + limit_order_book
        timestamps, one_day_basic_set, one_day_time_insensitive_set, one_day_labels = \
            extract_limit_order_book(limit_order_filename=limit_order_filename, feature_filename=feature_filepath)
        basic_set.extend(one_day_basic_set)
        time_insensitive_set.extend(one_day_time_insensitive_set)
        labels.extend(one_day_labels)

    basic_set = np.array(basic_set)
    time_insensitive_set = np.array(time_insensitive_set)
    labels = np.array(labels)

    train_index, test_index, idx = get_samples_index(labels, split = 0.25)
    features = np.concatenate((basic_set, time_insensitive_set), axis=1)
    selected_train_data = features[train_index]
    selected_train_labels = labels[train_index]
    print(train_index)
    selected_test_data = features[test_index]
    selected_test_labels = labels[test_index]
    print(test_index)

    # max_info_indices = feature_selection(selected_data, selected_labels)
    # selected_data = selected_data[:, max_info_indices]
    C = [5, 10, 20, 40]
    G = [1e-5, 1e-6, 1e-7, 1e-8]
    # optimal 5e4, 1e-7
    for c in C:
        for g in G:
            print("SVM c = {}".format(c) + " g = {}".format(g))
            score, model = svm_trainer.train_svm(train_data=selected_train_data, train_labels=selected_train_labels, \
                test_data=selected_test_data, test_labels=selected_test_labels,c=c, kernel='rbf', g=g)


def train_one_day(limit_order_filename, feature_filename,
                  n_level=10, time_interval=100):
    timestamps, basic_set, time_insensitive_set, labels = extract_limit_order_book(
        limit_order_filename=limit_order_filename, feature_filename=feature_filename,
        time_interval=time_interval, n_level=n_level)

    train_index, test_index, idx = get_samples_index(labels, split = 0.25)
    # selected_data = basic_set[sampling_index]
    # selected_data = time_insensitive_set[sampling_index]
    features = np.concatenate((basic_set, time_insensitive_set), axis=1)
    selected_train_data = features[train_index]
    selected_train_labels = labels[train_index]
    print(train_index)
    selected_test_data = features[test_index]
    selected_test_labels = labels[test_index]
    print(test_index)

    # max_info_indices = feature_selection(selected_data, selected_labels)
    # selected_data = selected_data[:, max_info_indices]
    C = [5, 5.5, 6, 6.5, 7, 7.5, 8]
    #, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5]
    G = [1e-7]
    # optimal 5e4, 1e-7
    for c in C:
        for g in G:
            print("SVM c = {}".format(c) + " g = {}".format(g))
            score, model = svm_trainer.train_svm(train_data=selected_train_data, train_labels=selected_train_labels, \
                test_data=selected_test_data, test_labels=selected_test_labels,c=c, kernel='rbf', g=g)

    # execution strategy based on ground truth
    test_data = features[idx:]
    test_labels = labels[idx:]
    cash = execution_strategy.execution(full_test_data=test_data, full_test_labels=test_labels, max_holdings=200, unit=1, tick_increment=0.01)



def extract_limit_order_book(limit_order_filename, feature_filename,
                             time_interval=100, n_level=10):
    extractor = feature_extractor.FeatureExtractor(
        limit_order_filename=limit_order_filename,
        feature_filename=feature_filename,
        time_interval=time_interval, n_level=n_level)
    timestamps, basic_set, time_insensitive_set, labels = extractor.extract_features()
    print("Order book {} has {} data points".format(limit_order_filename.split('/')[-1], len(labels)))
    return timestamps, basic_set, time_insensitive_set, labels
