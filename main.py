import ob_trainer
import sys
import os


# def train_order_book(input_dir, stock, date, event_time, label_type, n_level=10):
#     """Train one day's order book."""
#     limit_order_filename = os.path.join(input_dir, stock + "_OB_" + date + ".xlsx")
#     trd_order_filename = os.path.join(input_dir, stock + "_TRD_" + date + ".xlsx")
#     cancel_order_filename = os.path.join(input_dir, stock + "_Order_Cancel_" + date + ".xlsx")
#     submission_filename = os.path.join(input_dir, stock + "_SUB_" + date + ".xlsx")
#     # limit_order_filename = "../PN_0816/PN_OB_" + date + ".xlsx"
#     # trd_order_filename = "../PN_0816/PN_TRD_" + date + ".xlsx"
#     # cancel_order_filename = "../PN_0816/PN_Order_Cancel_" + date + ".xlsx"
#     # submission_filename = "../PN_0816/PN_SUB_" + date + ".xlsx"
#     feature_filename = "../output_zheng/" + stock + "_" + date + "_" + event_time + ".json"
#     ob_trainer.train_order_book(limit_order_filename, trd_order_filename,
#                                 cancel_order_filename, submission_filename,
#                                 feature_filename, event_time, label_type,
#                                 n_level=n_level, time_interval=100)


# def train_order_books(input_dir, stock, dates, event_time, label_type):
#     """Train order books."""
#     ob_trainer.train_order_books(input_dir, stock, dates, event_time, label_type,
#                                  time_interval=100, n_level=10)


if __name__ == '__main__':
    # python main.py ../PN_0816/ PN 081016,081116 E midspread
    input_dir = sys.argv[1]
    stock = sys.argv[2]
    dates = sys.argv[3].replace(' ', '').split(',')
    event_time = sys.argv[4]
    label_type = sys.argv[5]
    # train_order_books(input_dir, stock, dates, event_time, label_type)
    ob_trainer.train_order_books(input_dir, stock, dates, event_time, label_type,
                                 time_interval=100, n_level=10)