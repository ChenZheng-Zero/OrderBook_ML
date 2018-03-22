import pandas as pd
import  sys
import numpy as np
import os
import pdb


def detect_cancel(value):
    """
    Find orders that are cancelled or transacted
    """
    length = value.shape[0] # the length of the group
    cancel_df = pd.DataFrame(columns=value.columns)
    for i in range(1, length):
        if value.iloc[i]['SIZE'] < value.iloc[i-1]['SIZE']:
            tmp_order = value.iloc[i]
            tmp_order['SIZE'] = value.iloc[i-1]['SIZE'] - value.iloc[i]['SIZE']
            cancel_df = cancel_df.append(tmp_order)
    return cancel_df


def merge_cancel(value):
    """
    For given Timestamp and price, merge the sizes of all the orders
    """
    cancel_merge_df = pd.DataFrame(columns=value.columns)
    cancel_merge_df = cancel_merge_df.append(value.iloc[0])
    cancel_merge_df.iloc[0]['SIZE'] = value['SIZE'].sum()
    return cancel_merge_df


def get_cancel_order(order_filename):
    """
    get cancelled orders from 'order_filename', save these orders in tgt_path
    """
    src_path = os.path.join('../PN_0816/'+order_filename)
    tgt_path = os.path.join('../PN_0816/'+order_filename.replace('Raw', 'Cancel'))
    trd_path = os.path.join('../PN_0816/'+order_filename.replace('Order_Raw', 'TRD'))

    example = pd.read_excel(src_path)
    trade = pd.read_excel(trd_path).reindex(columns=['Time', 'SIZE', 'PRICE']).set_index(keys=['Time', 'PRICE'])

    exm_cancel = example.groupby(['ORDER_ID'], as_index=False)
    cancel_trd = exm_cancel.apply(detect_cancel) \
                .reset_index().drop(columns=['level_0', 'level_1'])
    cancel_mrg = cancel_trd.groupby(['Time', 'PRICE'], as_index=False).apply(merge_cancel) \
                .reset_index().drop(columns=['level_0', 'level_1'])

    cancel_ord = cancel_mrg.join(trade, on=['Time','PRICE'], rsuffix='_trd')\
                .fillna(value={'SIZE_trd': 0})
    cancel_ord['SIZE'] = cancel_ord['SIZE'] - cancel_ord['SIZE_trd']
    cancel_ord.drop(columns='SIZE_trd', inplace=True)
    cancel_ord = cancel_ord.loc[cancel_ord['SIZE'] > 0, :]
    # pdb.set_trace()
    cancel_ord.to_excel(tgt_path, index=False)


if __name__ == '__main__':
    get_cancel_order("PN_Order_Raw_" + sys.argv[1] + ".xlsx")
