from feature_extractor import get_delimiter_index
import pandas as pd


def check_level(limit_order_filename, n_level=10):
    limit_order_df = pd.read_excel(limit_order_filename)
    delimiter_index = get_delimiter_index(limit_order_df)
    guaranteed_index = 0
    for i in range(len(delimiter_index) - 1):
        count = 0
        if delimiter_index[i + 1] - delimiter_index[i] < n_level:
            guaranteed_index = i + 1
            continue
        for index in range(delimiter_index[i] + 1, delimiter_index[i + 1]):
            if limit_order_df["BID_SIZE"][index] * limit_order_df["ASK_SIZE"][index] > 0:
                count += 1
            else:
                if count < n_level:
                    print("{} contains insufficient orders".format(limit_order_df["Time"][index]))
                    guaranteed_index = i+1
    print("From {}, it's sufficient, timestamp is {}".format(delimiter_index[guaranteed_index],
                                                             limit_order_df["Time"][delimiter_index[guaranteed_index]]))


if __name__ == '__main__':
    check_level(limit_order_filename="../data/input/PN_OB_Snapshot_Aug10.xlsx")
