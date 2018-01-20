from pre_processor import get_samples_index
import feature_extractor
import numpy as np
import svm_trainer


if __name__ == '__main__':
    timestamps, basic_set, time_insensitive_set, time_sensitive_set, labels = \
        feature_extractor.extract_features(
            limit_order_filename="../../data/input/PN_OB_Snapshot_Aug10.xlsx",
            transaction_order_filename="../../data/input/PN_TRD_0810.xls",
            feature_filename="../../data/output/features_10.json",
            n_level=10, delta_t=50, delta_T=1000)
    sampling_index = get_samples_index(labels, num_per_label=900) # could add parameter for each type labels
    selected_data = basic_set[sampling_index]
    # selected_data = time_sensitive_set[sampling_index]
    # features = np.concatenate((time_insensitive_set, time_sensitive_set), axis=1)
    # selected_data = features[sampling_index]
    selected_labels = labels[sampling_index]
    svm_trainer.train_svm(data=selected_data, labels=selected_labels, c=1.0, kernel='rbf')

