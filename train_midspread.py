from mid_spread_label import get_mid_spread_labels
import pandas as pd
import numpy as np
from pre_processor import get_samples_index
import execution_strategy
import svm_trainer
import matplotlib.pyplot as plt


if __name__ == '__main__':
    filename_prefix = "../output_zheng/081616_E_10_midspread"
    thresholds = [0.25, 0.5, 0.75, 1]
    profits = []
    for threshold in thresholds:
        feature_filename = filename_prefix + "_" + str(threshold) + ".json"
        get_mid_spread_labels(feature_filename, threshold=threshold)
        df = pd.read_json(feature_filename, orient="records", lines="True")
        timestamps = np.array(df["timestamps"].tolist())
        basic_set = np.array(df["basic_set"].tolist())
        time_insensitive_set = np.array(df["time_insensitive_set"].tolist())
        time_sensitive_set = np.array(df["time_sensitive_set"].tolist())
        labels = np.array(df["mid_spread_labels"].tolist())

        train_index, test_index, idx = get_samples_index(labels, split=0.25)
        features = np.concatenate((basic_set, time_insensitive_set, time_sensitive_set), axis=1)
        selected_train_data = features[train_index]
        selected_train_labels = labels[train_index]
        selected_test_data = features[test_index]
        selected_test_labels = labels[test_index]

        # max_info_indices = feature_selection(selected_data, selected_labels)
        # selected_data = selected_data[:, max_info_indices]
        # C = [1, 1e1, 1e2, 1e3, 1e4, 1e5]
        # G = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
        # # optimal 5e4, 1e-7
        # # optimal 10000, 1e-8, accuracy 78%, precision 71%
        # for c in C:
        #     for g in G:
        #         print("SVM c = {}".format(c) + " g = {}".format(g))
        #         score, model = svm_trainer.train_svm(train_data=selected_train_data, train_labels=selected_train_labels, \
        #                                              test_data=selected_test_data, test_labels=selected_test_labels, c=c, kernel='rbf', g=g)

        # execution strategy based on ground truth
        test_data = features[idx:]
        test_labels = labels[idx:]
        cash = execution_strategy.execution(full_test_data=test_data, full_test_labels=test_labels,
                                            max_holdings=200, unit=1, tick_increment=0.01)
        profits.append(cash)
    plt.plot(thresholds, profits, 'ro')
    plt.xlabel("thresholds")
    plt.ylabel("cash")
    plt.show()

