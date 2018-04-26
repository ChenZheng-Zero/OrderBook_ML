from utils import time_to_int, int_to_time
import numpy as np
import os
import pandas as pd
import pdb


class FeatureExtractor:

    def __init__(self, limit_order_filename, trd_filename, cancel_order_filename,
                 submission_filename, feature_filename, event_time,
                 time_interval, n_level):
        self.limit_order_filename = limit_order_filename
        self.trd_filename = trd_filename
        self.cancel_order_filename = cancel_order_filename
        self.submission_filename = submission_filename
        self.limit_order_df = None
        self.trd_df = None
        self.feature_filename = feature_filename
        self.event_time = event_time
        self.time_interval = time_interval
        self.n_level = n_level
        self.delimiter_indices = []
        self.indices = []

    def extract_features(self):
        """Extract features from limit order book."""
        if not os.path.isfile(self.feature_filename):
            self.limit_order_df = pd.read_excel(self.limit_order_filename)
            self.trd_df = pd.read_excel(self.trd_filename).sort_values('Time').reset_index()
            
            self.delimiter_indices = self.get_delimiter_indices()
            if self.event_time == 'E':
                self.indices = range(0, len(self.delimiter_indices))
            else:
                self.indices = self.get_time_interval_indices()

            basic_set, timestamps, mid_prices, max_mid_prices = self.extract_basic_set()
            mid_price_labels = self.get_mid_price_labels(mid_prices, max_mid_prices)
            spread_crossing_labels = self.get_spread_crossing_labels(np.array(basic_set)[:, 2], np.array(basic_set)[:, 0])
            # pdb.set_trace()
            time_insensitive_set = self.extract_time_insensitive_set(basic_set)
            time_sensitive_set = self.extract_time_sensitive_set(timestamps)
            self.save_feature_json(self.feature_filename, timestamps, basic_set,
                                   time_insensitive_set, mid_price_labels, spread_crossing_labels, mid_prices, max_mid_prices)

        df = pd.read_json(self.feature_filename, orient="records", lines="True", convert_dates = False)
        timestamps = df["timestamps"].tolist()
        basic_set = df["basic_set"].tolist()
        time_insensitive_set = df["time_insensitive_set"].tolist()
        mid_price_labels = df["mid_price_labels"].tolist()
        spread_crossing_labels = df["spread_crossing_labels"].tolist()
        return np.array(timestamps), np.array(basic_set), \
            np.array(time_insensitive_set), np.array(mid_price_labels), np.array(spread_crossing_labels)

    def extract_basic_set(self):
        """Extract basic set."""
        limit_book_indices = np.array(self.delimiter_indices)[self.indices].tolist()
        assert(len(limit_book_indices) > 0)
        timestamps = []
        basic_set = []
        mid_prices = []
        max_mid_prices = []
        if self.event_time == 'T':
            init_index = 0
            init_time = self.get_init_time(limit_book_indices)

        trd_i = 0
        for i in limit_book_indices:
            # append the timestamp
            if self.event_time == 'E':
                timestamps.append(time_to_int(self.limit_order_df["Time"][i+1]))
            else:
                timestamps.append(init_time)
            
            # append basic features       
            v1 = []
            for index in range(i + 1, i + 1 + self.n_level):
                v1.append([self.limit_order_df["ASK_PRICE"][index],
                           self.limit_order_df["ASK_SIZE"][index],
                           self.limit_order_df["BID_PRICE"][index],
                           self.limit_order_df["BID_SIZE"][index]])
            basic_set.append(np.array(v1).reshape(1, -1).tolist()[0])

            # append mid-price (if a trade, trade price; else mid price of the snapshot)
            is_trd = 0
            while trd_i < self.trd_df.shape[0]-1 and time_to_int(self.trd_df['Time'][trd_i]) < timestamps[-1]:
                trd_i = trd_i+1

            temp = trd_i
            while trd_i < self.trd_df.shape[0]-1 and time_to_int(self.trd_df['Time'][trd_i]) == timestamps[-1]:
                # incoming buy order transacted with an incumbent sell
                if self.trd_df['PRICE'][trd_i] == basic_set[-2][0]:
                    # partial transaction
                    if self.trd_df['PRICE'][trd_i] == basic_set[-1][0] and self.trd_df['SIZE'][trd_i] == basic_set[-2][1]-basic_set[-1][1]:
                        #print("partial sell transaction at time {} with price {}".format(int_to_time(timestamps[-1]), self.trd_df['PRICE'][trd_i]))
                        is_trd = 1
                        break
                    # full transaction    
                    elif self.trd_df['PRICE'][trd_i] != basic_set[-1][0] and self.trd_df['SIZE'][trd_i] == basic_set[-2][1]: 
                        #print("full sell transaction at time {} with price {}".format(int_to_time(timestamps[-1]), self.trd_df['PRICE'][trd_i]))
                        is_trd = 1
                        break
                # incoming sell order transacted with an incumbent buy
                elif self.trd_df['PRICE'][trd_i] == basic_set[-2][2]:
                    # partial transaction
                    if self.trd_df['PRICE'][trd_i] == basic_set[-1][2] and self.trd_df['SIZE'][trd_i] == basic_set[-2][3]-basic_set[-1][3]:
                        #print("partial buy transaction at time {} with price {}".format(int_to_time(timestamps[-1]), self.trd_df['PRICE'][trd_i]))
                        is_trd = 1
                        break
                    # full transaction
                    elif self.trd_df['PRICE'][trd_i] != basic_set[-1][2] and self.trd_df['SIZE'][trd_i] == basic_set[-2][3]: 
                        #print("full buy transaction at time {} with price {}".format(int_to_time(timestamps[-1]), self.trd_df['PRICE'][trd_i]))
                        is_trd = 1
                        break
                trd_i = trd_i+1

            if is_trd == 1:
                #pdb.set_trace()
                mid_price = self.trd_df['PRICE'][trd_i]
            else:
                mid_price = (self.limit_order_df["ASK_PRICE"][i+1]\
                        + self.limit_order_df["BID_PRICE"][i+1])/2
            trd_i = temp

            mid_prices.append(mid_price)
            max_mid_price = mid_price

            if self.event_time == 'T':
                while init_index < len(self.delimiter_indices) and np.array(self.delimiter_indices)[init_index] <= i:
                    max_mid_price = max(max_mid_price, (self.limit_order_df["ASK_PRICE"][np.array(self.delimiter_indices)[init_index]+1]
                        + self.limit_order_df["BID_PRICE"][np.array(self.delimiter_indices)[init_index]+1])/2)
                    init_index = init_index + 1
                # update the snapshot timestamp
                init_time = init_time + self.time_interval

            max_mid_prices.append(max_mid_price)  

        # reach the end of trading period   
        max_mid_price = (self.limit_order_df["ASK_PRICE"][np.array(self.delimiter_indices)[-1]+1]\
                + self.limit_order_df["BID_PRICE"][np.array(self.delimiter_indices)[-1]+1])/2
        if self.event_time == 'T':
            while init_index < len(self.delimiter_indices) and np.array(self.delimiter_indices)[init_index] <= self.delimiter_indices[-1]:
                max_mid_price = max(max_mid_price, (self.limit_order_df["ASK_PRICE"][np.array(self.delimiter_indices)[init_index]+1]\
                    + self.limit_order_df["BID_PRICE"][np.array(self.delimiter_indices)[init_index]+1])/2)
                init_index = init_index + 1

        max_mid_prices.append(max_mid_price) 
        max_mid_prices = max_mid_prices[1:]

        return basic_set, timestamps, mid_prices, max_mid_prices

    def extract_time_insensitive_set(self, basic_set):
        """Extract time insensitive features."""
        time_insensitive_set = []
        for v1 in basic_set:
            v1 = np.array(v1).reshape(self.n_level, -1)
            v2 = self.get_time_insensitive_v2(v1)
            v3 = self.get_time_insensitive_v3(v1)
            v4 = self.get_time_insensitive_v4(v1)
            v5 = self.get_time_insensitive_v5(v1)
            time_insensitive_feature = v2 + v3 + v4 + v5
            time_insensitive_set.append(time_insensitive_feature)
        return time_insensitive_set

    def get_delimiter_indices(self):
        """Get all valid D's indices in the limit order book"""
        delimiter_indices = [-1] # assume there is a D before index 0
        for i in range(len(self.limit_order_df["DELIMITER"])):
            if self.limit_order_df["DELIMITER"][i] == "D":
                delimiter_indices.append(i)
        guarantee_index = self.check_n_level(delimiter_indices)
        return delimiter_indices[guarantee_index:-1]

    def check_n_level(self, delimiter_indices):
        """Find the first index in delimiter indices that satisfies the desired level."""
        guaranteed_index = 0
        for i in range(len(delimiter_indices) - 1):
            count = 0
            if delimiter_indices[i + 1] - delimiter_indices[i] < self.n_level:
                guaranteed_index = i + 1
                continue
            for index in range(delimiter_indices[i] + 1, delimiter_indices[i + 1]):
                if self.limit_order_df["BID_SIZE"][index] * self.limit_order_df["ASK_SIZE"][index] > 0:
                    count += 1
                    if count == self.n_level:
                        break
                else:
                    if count < self.n_level:
                        guaranteed_index = i + 1
        # print("guaranteed_index: ", guaranteed_index)
        return guaranteed_index

    def get_time_interval_indices(self):
        """Find all D's indices in delimiter indices for time intervals."""
        next_timestamp = self.get_start_timestamp()
        time_interval_indices = []
        current_index = 0
        while current_index != -1:
            time_interval_index = self.get_time_interval_index(next_timestamp, current_index)
            if time_interval_index == -1:
                break
            time_interval_indices.append(time_interval_index)
            current_index = time_interval_index
            next_timestamp += self.time_interval
        return time_interval_indices

    def get_time_interval_index(self, timestamp, current_index):
        """Find the first state for the desired timestamp."""
        for i in range(current_index, len(self.delimiter_indices)):
            index = self.delimiter_indices[i] + 1
            current_timestamp = time_to_int(self.limit_order_df["Time"].iloc[index])
            if current_timestamp == timestamp:
                return i
            elif current_timestamp > timestamp:
                return i - 1
        return -1

    def get_start_timestamp(self):
        """Find the first timestamp in int."""
        assert(len(self.delimiter_indices) > 0)
        guaranteed_index = self.delimiter_indices[0] + 1
        # print("Start timestamp:", self.limit_order_df["Time"].iloc[guaranteed_index])
        guaranteed_timestamp = time_to_int(self.limit_order_df["Time"].iloc[guaranteed_index])
        if guaranteed_timestamp % self.time_interval == 0:
            start_timestamp = guaranteed_timestamp
        else:
            start_timestamp = self.time_interval * ((guaranteed_timestamp / self.time_interval) + 1)
        # print("Start timestamp int:", start_timestamp)
        return start_timestamp

    def get_init_time(self, limit_book_indices):
        if limit_book_indices[0] == -1:
            init_time_index = 0
        else:
            init_time_index = limit_book_indices[0]
        init_time = time_to_int(self.limit_order_df["Time"][init_time_index]) \
            if time_to_int(self.limit_order_df["Time"][init_time_index]) % self.time_interval == 0 \
            else self.time_interval * ((time_to_int(self.limit_order_df["Time"][init_time_index])
                                        / self.time_interval) + 1)
        return init_time

    @staticmethod
    def get_time_insensitive_v2(v1):
        """Get v2 from v1."""
        v2 = [[v1_i[0] - v1_i[2], (v1_i[0] + v1_i[2])/2] for v1_i in v1]
        return [var for v2_i in v2 for var in v2_i]

    @staticmethod
    def get_time_insensitive_v3(v1):
        """Get v3 from v1."""
        v3 = [[v1[-1][0] - v1[0][0], v1[0][2] - v1[-1][2],
               abs(v1[i][0] - v1[i - 1][0]), abs(v1[i][2] - v1[i - 1][2])]
              for i in range(len(v1)) if i > 0]
        return [var for v3_i in v3 for var in v3_i]

    @staticmethod
    def get_time_insensitive_v4(v1):
        """Get v4 from v1."""
        p_ask = [v1_i[0] for v1_i in v1]
        v_ask = [v1_i[1] for v1_i in v1]
        p_bid = [v1_i[2] for v1_i in v1]
        v_bid = [v1_i[3] for v1_i in v1]
        return [sum(p_ask)/len(p_ask), sum(p_bid)/len(p_bid),
                sum(v_ask)/len(v_ask), sum(v_bid)/len(v_bid)]

    @staticmethod
    def get_time_insensitive_v5(v1):
        """Get v5 from v1."""
        p_ask_p_bid = [v1_i[0] - v1_i[2] for v1_i in v1]
        v_ask_v_bid = [v1_i[1] - v1_i[3] for v1_i in v1]
        return [sum(p_ask_p_bid), sum(v_ask_v_bid)]

    @staticmethod
    def get_mid_price_labels(mid_prices, max_mid_prices):
        """Get the mid price labels"""
        gt = []
        for i in range(0, len(mid_prices)):
            if max_mid_prices[i] - mid_prices[i] > 0:
                gt.append(1)
            elif max_mid_prices[i] - mid_prices[i] < 0:
                gt.append(-1)
            else:
                gt.append(0)
        return gt

    @staticmethod
    def get_spread_crossing_labels(best_bids, best_asks):
        """Get the spread crossing labels"""
        gt = []
        for i in range(0, len(best_asks)-1):
            if best_bids[i+1] - best_asks[i] >= 0:
                gt.append(1)
            else:
                gt.append(0)
        gt.append(0)
        return gt

    @staticmethod
    def save_feature_json(feature_filename, timestamps, basic_set,
                          time_insensitive_set, mid_price_labels, spread_crossing_labels, mid_prices, max_mid_prices):
        """Save the json."""
        feature_dict = {"timestamps": timestamps, "basic_set": basic_set,
                        "time_insensitive_set": time_insensitive_set,
                        "mid_price_labels": mid_price_labels, "spread_crossing_labels": spread_crossing_labels,
                        "mid_prices": mid_prices, "max_mid_prices": max_mid_prices}
        df = pd.DataFrame(data=feature_dict, columns=["timestamps", "basic_set",
                                                      "time_insensitive_set",
                                                      "mid_price_labels", "spread_crossing_labels",
                                                      "mid_prices", "max_mid_prices"])
        df.to_json(path_or_buf=feature_filename, orient="records", lines=True)


