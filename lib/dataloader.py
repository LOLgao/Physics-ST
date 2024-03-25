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


def STDataloader(x, y, date, event, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    x, y, date, event = TensorFloat(x), TensorFloat(y), TensorFloat(date), TensorFloat(event)
    data = torch.utils.data.TensorDataset(x, y, date, event)
    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


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
        data['date_' + category] = cat_data['date']
        data['event_' + category] = cat_data['event']

    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), scalar_type)
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
    # Construct dataloader
    train_dataloader = STDataloader(
        data['x_train'],
        data['y_train'],
        data['date_train'],
        data['event_train'],
        batch_size,
        shuffle=True
    )
    val_dataloader = STDataloader(
        data['x_val'],
        data['y_val'],
        data['date_val'],
        data['event_val'],
        batch_size,
        shuffle=False
    )
    test_dataloader = STDataloader(
        data['x_test'],
        data['y_test'],
        data['date_test'],
        data['event_test'],
        batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader, scaler
