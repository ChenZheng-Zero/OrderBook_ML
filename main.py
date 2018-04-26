import ob_trainer
import sys
import os
import pdb

# @click.command()
# @click.argument('date')
# @click.argument('event_time')
# @click.argument('label_type')
# def train_one_day(date, event_time, label_type, n_level=10):
#     """Train one day's order book."""
#     limit_order_filename = "../GOOG_0817/GOOG_OB_" + date + ".xlsx"
#     trd_filename = "../GOOG_0817/GOOG_TRD_" + date[:-1] + ".xlsx"
#     feature_filename = "../output/" + date + "_" + str(n_level) + ".json"
#     ob_trainer.train_one_day(limit_order_filename, trd_filename, feature_filename, event_time, label_type,
#                              n_level=10, time_interval=100)
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