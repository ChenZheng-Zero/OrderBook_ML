import pandas as pd
import os


def detect_submission(value):
    """Detect submission orders."""
    # the length of the group
    order_dict = {}
    length = value.shape[0]
    submission_df = pd.DataFrame(columns=value.columns)
    for i in range(1, length):
        if value.iloc[i]['SIZE'] > 0:
            if value.iloc[i]['PRICE'] not in order_dict:
                submission_df = submission_df.append(value.iloc[i])
            elif value.iloc[i]['SIZE'] > order_dict[value.iloc[i]['PRICE']]:
                tmp_order = value.iloc[i]
                tmp_order['SIZE'] = value.iloc[i]['SIZE'] - order_dict[value.iloc[i]['PRICE']]
                submission_df = submission_df.append(tmp_order)
        order_dict[value.iloc[i]['PRICE']] = value.iloc[i]['SIZE']
    return submission_df


def get_submission_order(order_filename):
    """Get submission orders from raw data."""
    src_path = os.path.join('../PN_0816/', order_filename)
    tgt_path = src_path.replace('Order_Raw', 'SUB')

    raw_data = pd.read_excel(src_path)
    data = raw_data.groupby(['ORDER_ID'], as_index=False)
    submission_ord = data.apply(detect_submission).reset_index().drop(columns=['level_0', 'level_1'])
    submission_ord.to_excel(tgt_path, index=False)


if __name__ == '__main__':
    get_submission_order("PN_Order_Raw_080116.xlsx")