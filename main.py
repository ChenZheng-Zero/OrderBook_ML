import ob_trainer


def train_one_day():
    """Train one day's order book."""
    n_level = 10
    limit_order_filename = "../../data/input/OB/PN_OB_081016.xlsx"
    feature_filename = "../../data/output/081016_" + str(n_level) + ".json"
    ob_trainer.train_one_day(limit_order_filename, feature_filename, num_per_label=2900,
                             n_level=10, time_interval=100)


def train_all_order_books():
    """Train all order books."""
    n_level = 10
    input_folder = "../../data/input/OB/"
    output_folder = "../../data/output/"
    ob_trainer.train_all_days(input_folder, output_folder, n_level=n_level)


if __name__ == '__main__':
    train_one_day()
