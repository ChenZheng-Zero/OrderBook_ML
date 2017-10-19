from pre_processor import get_samples_index
import feature_extractor
import svm_trainer


if __name__ == '__main__':
    timestamps, basic_set, time_insensitive_set, labels = \
        feature_extractor.extract_features_from_order_books(
            limit_order_filename="../../data/input/PN_OB_Snapshot_Aug10.xlsx",
            feature_filename="../../data/output/features_d_10.json", n_level=10)
    sampling_index = get_samples_index(labels, num_per_label=700) # could add parameter for each type labels
    selected_data = basic_set[sampling_index]
    # selected_data = time_insensitive_set[sampling_index]
    selected_labels = labels[sampling_index]
    svm_trainer.train_svm(data=selected_data, labels=selected_labels, c=1.0, kernel='rbf')


