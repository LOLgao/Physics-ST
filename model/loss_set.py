import torch
import torch.nn as nn
from lib.metrics import MAE_torch


def masked_mae_loss(mask_value):
    def loss(preds, labels):
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss
