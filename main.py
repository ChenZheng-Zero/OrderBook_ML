from pre_processor import get_samples_index
import feature_extractor
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import svm_trainer
import pdb


if __name__ == '__main__':
    #n_levels = [5, 7, 10]
    n_levels = [10]
    for n_level in n_levels:
        print("level:{}".format(n_level))
        extractor = feature_extractor.FeatureExtractor(
            limit_order_filename="../input/PN_OB_Snapshot_Aug10.xlsx",
            feature_filename="../output/features_" + str(n_level) + ".json",
            time_interval=100, n_level=n_level)
        timestamps, basic_set, time_insensitive_set, labels = extractor.extract_features()
        sampling_index = get_samples_index(labels, num_per_label=2950)
        # selected_data = basic_set[sampling_index]
        # selected_data = time_insensitive_set[sampling_index]
        features = np.concatenate((basic_set, time_insensitive_set), axis=1)
        #pdb.set_trace()
        #mutual_info_classif(features, labels)
        selected_data = features[sampling_index]
        selected_labels = labels[sampling_index]
        C = [0.1, 1, 10, 100, 1000]
        G = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
        for c in C:
            for g in G:
                print("SVM c = {}".format(c) + " g = {}".format(g))
                svm_trainer.train_svm(data=selected_data, labels=selected_labels, c=c, kernel='rbf', g=g)

