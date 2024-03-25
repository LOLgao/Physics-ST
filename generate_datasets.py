import numpy as np
import pandas as pd
import os

def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets):
    num_samples, num_nodes = data.shape
    data = np.expand_dims(data.values, axis=-1)

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

if __name__ == "__main__":
    df = pd.read_csv('data/ode_data/D1/raw_data.csv', index_col=[0])  ###########
    pre_len = 24
    output_dir = 'data/ode_data/D1/'
    num_samples = df.shape[0]
    x_offsets = np.arange(-23, 1, 1)
    y_offsets = np.arange(1, pre_len+1, 1)
    num_train = int(num_samples * 0.6)
    num_val = int(num_samples * 0.8)

    x_train, y_train = generate_graph_seq2seq_io_data(df[:num_train], x_offsets, y_offsets)
    x_val, y_val = generate_graph_seq2seq_io_data(df[num_train:num_val], x_offsets, y_offsets)
    x_test, y_test = generate_graph_seq2seq_io_data(df[num_val:], x_offsets, y_offsets)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()['y_' + cat]
        _date = np.load(output_dir + 'Temporal/' + cat + '/date.npy')
        _event = np.load(output_dir + 'Temporal/' + cat + '/event.npy')
        np.savez_compressed(
            os.path.join(output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            date=_date,
            event=_event)