import os
import time
import torch
import numpy as np


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


class MinMax01Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)


class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


# def STDataloader(X, Y, batch_size, shuffle=True, drop_last=True):
#     cuda = True if torch.cuda.is_available() else False
#     TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#     X, Y = TensorFloat(X), TensorFloat(Y)
#     data = torch.utils.data.TensorDataset(X, Y)
#     dataloader = torch.utils.data.DataLoader(
#         data,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=drop_last,
#     )
#     return dataloader
def STDataloader(x, y, x_condition, x_disease, x_distance, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    x, y, x_condition = TensorFloat(x), TensorFloat(y), TensorFloat(x_condition)
    x_disease, x_distance = TensorFloat(x_disease), TensorFloat(x_distance)
    data = torch.utils.data.TensorDataset(x, y, x_condition, x_disease)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader, x_distance


def normalize_data(data, scalar_type='Standard'):
    scalar = None
    if scalar_type == 'Standard':
        scalar = StandardScaler(mean=data.mean(), std=data.std())
    else:
        raise ValueError('scalar_type is not supported in data_normalization.')
    # print('{} scalar is used!!!'.format(scalar_type))
    # time.sleep(3)
    return scalar


def get_dataloader(data_dir, batch_size, scalar_type='Standard'):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['x_condition_' + category] = cat_data['x_date']
        data['x_disease_' + category] = cat_data['x_disease']
        data['x_distance_' + category] = cat_data['x_ditance']
    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), scalar_type)
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
    # Construct dataloader
    train_dataloader, distance = STDataloader(
        data['x_train'],
        data['y_train'],
        data['x_condition_train'],
        data['x_disease_train'],
        data['x_distance_train'],
        batch_size,
        shuffle=True
    )
    val_dataloader, _ = STDataloader(
        data['x_val'],
        data['y_val'],
        data['x_condition_val'],
        data['x_disease_val'],
        data['x_distance_val'],
        batch_size,
        shuffle=False
    )
    test_dataloader, _ = STDataloader(
        data['x_test'],
        data['y_test'],
        data['x_condition_test'],
        data['x_disease_test'],
        data['x_distance_test'],
        batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader, distance, scaler
