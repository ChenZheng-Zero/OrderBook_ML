import ob_trainer
import sys
import os
import pdb

if __name__ == '__main__':
    # python3 main.py ../GOOG_0817/ GOOG $date A E $thresh 2 > ../results_zheng/GOOG_$date_$thresh_midspread.txt
    input_dir = sys.argv[1]
    stock = sys.argv[2]
    dates = sys.argv[3].replace(' ', '').split(',')
    # M(morning) or A(afternoon) or W(wholeday)
    span_type = sys.argv[4]
    # E(event-based) or T(time interval based) 
    event_time = sys.argv[5]
    label_type = sys.argv[6]
    feature_type = sys.argv[7]

    timestamps, basic_set, time_insensitive_set, time_sensitive_set, labels = ob_trainer.extract_features_from_order_books(input_dir, stock, dates, span_type, event_time, label_type,
                                 time_interval=100, n_level=10)
    ob_trainer.train_model_test_strategy(timestamps=timestamps, basic_set=basic_set, time_insensitive_set=time_insensitive_set, time_sensitive_set=time_sensitive_set, feature_level=feature_type, labels=labels, dates=dates)