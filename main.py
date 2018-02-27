import ob_trainer


def train_one_day():
    """Train one day's order book."""
    n_level = 10
    limit_order_filename = "../PN_0816/PN_OB_080116.xlsx"
    feature_filename = "../output/080116_" + str(n_level) + ".json"
    ob_trainer.train_one_day(limit_order_filename, feature_filename, n_level=10, time_interval=100)


def train_all_order_books():
    """Train all order books."""
    n_level = 10
    input_folder = "../GOOGL/"
    output_folder = "../output/"
    ob_trainer.train_all_days(input_folder, output_folder, n_level=n_level)


if __name__ == '__main__':
    train_all_order_books()
