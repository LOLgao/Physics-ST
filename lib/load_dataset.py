import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset):
    #output B, N, D
    print(dataset)
    if dataset == 'hfy':
        data_path = os.path.join('../data/hfy_0912_time_enc.csv')
        data = pd.read_csv(data_path, index_col=0)  #onley the first dimension, traffic flow data
        data = data.iloc[:, 1:].values
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
