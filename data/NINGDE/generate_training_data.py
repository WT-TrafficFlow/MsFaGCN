from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    # data = np.expand_dims(df.values, axis=-1)
    data = np.reshape(df.values[:,-1:],newshape=[-1, 30, 1]) # [N, nodes, 1]
    num_samples, num_nodes, _ = data.shape
    data_list = [data]
    if add_time_in_day:
        df['datatime'] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h') + pd.to_timedelta(
            df.minute,
            unit='m')

        time_ind = (df['datatime'].values - df['datatime'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        # time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        time_in_day = time_ind
        time_in_day = time_in_day.reshape(-1, 30, 1)

        data_list.append(time_in_day)

    if add_day_in_week:
        df['datatime'] = pd.to_datetime(df.date) + pd.to_timedelta(df.hour, unit='h') + pd.to_timedelta(
            df.minute,
            unit='m')
        df['datatime'] = pd.to_datetime(df['datatime'])
        day_of_week = df['datatime'].dt.dayofweek.to_numpy()
        day_of_weeks = np.reshape(day_of_week, newshape=[-1, num_nodes, 1])

        # day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        # day_in_week[np.arange(num_samples)[:, None], np.arange(num_nodes), day_of_weeks] = 1
        data_list.append(day_of_weeks)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    df = pd.read_csv(args.traffic_df_filename)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 13, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train


    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="train_baseline_5min.csv",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
