import pandas as pd
import os


def get_mid_spread_labels(filename, output_filename, threshold=0.3):
    df = pd.read_json(filename, orient="records", lines="True")
    timestamps = df["timestamps"].tolist()
    basic_set = df["basic_set"].tolist()
    time_insensitive_set = df["time_insensitive_set"].tolist()
    time_sensitive_set = df["time_sensitive_set"].tolist()
    mid_price_labels = df["mid_price_labels"].tolist()
    spread_crossing_labels = df["spread_crossing_labels"].tolist()
    mid_prices = df["mid_prices"].tolist()
    max_mid_prices = df["max_mid_prices"].tolist()
    mid_spread_labels = []

    for i in range(len(basic_set) - 1):
        spread = get_spread(basic_set[i])
        if spread <= 0:
            mid_spread_label = 1
        else:
            mid_spread_label = int((mid_prices[i+1] - mid_prices[i])/(spread/2) > threshold)
        mid_spread_labels.append(mid_spread_label)

    save_feature_json(output_filename, timestamps[:-1], basic_set[:-1],
                      time_insensitive_set[:-1], time_sensitive_set[:-1],
                      mid_price_labels[:-1], spread_crossing_labels[:-1],
                      mid_spread_labels, mid_prices[:-1], max_mid_prices[:-1])


def get_spread(v1):
    return v1[0] - v1[2]


def save_feature_json(feature_filename, timestamps, basic_set,
                      time_insensitive_set, time_sensitive_set,
                      mid_price_labels, spread_crossing_labels,
                      mid_spread_labels,
                      mid_prices, max_mid_prices):
    """Save the json."""
    feature_dict = {"timestamps": timestamps, "basic_set": basic_set,
                    "time_insensitive_set": time_insensitive_set,
                    "time_sensitive_set": time_sensitive_set,
                    "mid_price_labels": mid_price_labels, "spread_crossing_labels": spread_crossing_labels,
                    "mid_spread_labels": mid_spread_labels,
                    "mid_prices": mid_prices, "max_mid_prices": max_mid_prices}
    df = pd.DataFrame(data=feature_dict, columns=["timestamps", "basic_set",
                                                  "time_insensitive_set",
                                                  "time_sensitive_set",
                                                  "mid_price_labels", "spread_crossing_labels",
                                                  "mid_spread_labels",
                                                  "mid_prices", "max_mid_prices"])
    df.to_json(path_or_buf=feature_filename, orient="records", lines=True)


if __name__ == '__main__':
    files = ["../output_zheng/" + file for file in os.listdir("../output_zheng/") if file.endswith("E.json")]
    for filename in files:
        for threshold in [0.25, 0.5, 0.75, 1]:
            output_filename = filename[:-5] + "_midspread_" + str(threshold) + ".json"
            get_mid_spread_labels(filename, output_filename, threshold=threshold)