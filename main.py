import ob_trainer
import sys
import os


def train_one_day(date, event_time, label_type, n_level=10):
    """Train one day's order book."""
    limit_order_filename = "../PN_0816/PN_OB_" + date + ".xlsx"
    trd_order_filename = "../PN_0816/PN_TRD_" + date + ".xlsx"
    cancel_order_filename = "../PN_0816/PN_Order_Cancel_" + date + ".xlsx"
    submission_filename = "../PN_0816/PN_SUB_" + date + ".xlsx"
    feature_filename = "../output_zheng/" + date + "_" + event_time + '_' + str(n_level) + ".json"
    ob_trainer.train_one_day(limit_order_filename, trd_order_filename,
                             cancel_order_filename, submission_filename,
                             feature_filename, event_time, label_type,
                             n_level=10, time_interval=100)


# def train_all_order_books(event_time, label_type):
#     """Train all order books."""
#     input_folder = "../../data/input/OB/"
#     output_folder = "../../data/output/"
#     ob_trainer.train_all_days(input_folder, output_folder, event_time, label_type,
#                               time_interval=100, n_level=10)


if __name__ == '__main__':
    date = sys.argv[1]
    event_time = sys.argv[2]
    label_type = sys.argv[3]
    train_one_day(date, event_time, label_type)
