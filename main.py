from pre_processor import get_samples_index
import feature_extractor
import numpy as np
import pandas as pd
import svm_trainer


def read_features(feature_filename):
    df = pd.read_json(path_or_buf=feature_filename, orient="records", lines=True)
    timestamps = np.array(df["timestamps"].tolist())
    basic_set = np.array(df["basic_set"].tolist())
    time_insensitive_set = np.array(df["time_insensitive_set"].tolist())
    labels = np.array(df["labels"].tolist())
    return timestamps, basic_set, time_insensitive_set, labels


if __name__ == '__main__':
    timestamps, basic_set, time_insensitive_set, labels = read_features(feature_filename="./features.json")
    sampling_index = get_samples_index(labels, num_per_label=1000) # could add parameter for each type labels
    selected_data = basic_set[sampling_index]
    # selected_data = time_insensitive_set[sampling_index]
    selected_labels = labels[sampling_index]
    for c in [0.01, 0.1, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0, 100.0]:
        svm_trainer.train_svm(data=selected_data, labels=selected_labels, C=c)


