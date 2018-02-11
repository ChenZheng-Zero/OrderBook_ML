# from sklearn.feature_selection import mutual_info_classif
import ob_trainer


if __name__ == '__main__':
    input_folder = "../../data/input/OB/"
    output_folder = "../../data/output/"
    ob_trainer.train_all_days(input_folder, output_folder, n_level=10)

