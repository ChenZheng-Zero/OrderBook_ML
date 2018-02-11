from pre_processor import get_samples_index
from pre_processor import feature_selection
import feature_extractor
import numpy as np
import svm_trainer
import os
import utils


def train_all_days(input_folder, output_folder, n_level=10):
    limit_order_books = [file for file in os.listdir(input_folder)
                         if file.startswith("PN_OB")]
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
    sampling_index = get_samples_index(labels, num_per_label=30000)
    # selected_data = basic_set[sampling_index]
    # selected_data = time_insensitive_set[sampling_index]
    features = np.concatenate((basic_set, time_insensitive_set), axis=1)
    selected_data = features[sampling_index]
    selected_labels = labels[sampling_index]
    svm_trainer.train_svm(data=selected_data, labels=selected_labels, c=1.0, kernel='rbf', g=0.01)


def train_one_day(limit_order_filename, feature_filename, num_per_label,
                  n_level=10, time_interval=100):
    timestamps, basic_set, time_insensitive_set, labels = extract_limit_order_book(
        limit_order_filename=limit_order_filename, feature_filename=feature_filename,
        time_interval=time_interval, n_level=n_level)
    sampling_index = get_samples_index(labels, num_per_label=num_per_label)
    # selected_data = basic_set[sampling_index]
    # selected_data = time_insensitive_set[sampling_index]
    features = np.concatenate((basic_set, time_insensitive_set), axis=1)
    selected_data = features[sampling_index]
    selected_labels = labels[sampling_index]
    max_info_indices = feature_selection(selected_data, selected_labels)
    selected_data = selected_data[:, max_info_indices]
    svm_trainer.train_svm(data=selected_data, labels=selected_labels, c=1.0, kernel='rbf', g=0.01)


def extract_limit_order_book(limit_order_filename, feature_filename,
                             time_interval=100, n_level=10):
    extractor = feature_extractor.FeatureExtractor(
        limit_order_filename=limit_order_filename,
        feature_filename=feature_filename,
        time_interval=time_interval, n_level=n_level)
    timestamps, basic_set, time_insensitive_set, labels = extractor.extract_features()
    print("Order book {} has {} data points".format(limit_order_filename.split('/')[-1], len(labels)))
    return timestamps, basic_set, time_insensitive_set, labels
