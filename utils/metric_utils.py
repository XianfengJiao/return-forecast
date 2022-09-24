import os
import sys
import uuid
import random
import shutil
import errno
import requests
import yaml
import json
import logging
import datetime
import hashlib
import contextlib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from typing import Callable, Iterable, Optional, Tuple, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

matplotlib.use('Agg')

from glob import glob
from omegaconf import OmegaConf
from logging import StreamHandler, Handler, getLevelName

EPS = 1e-12



def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



class K(object):

    @staticmethod
    def mean(x, axis=0, keepdim=True):
        if isinstance(x, np.ndarray):
            return x.mean(axis=axis, keepdims=keepdim)
        if isinstance(x, torch.Tensor):
            return x.mean(dim=axis, keepdim=keepdim)
        raise NotImplementedError('upsupport data type %s'% type(x))
    
    @staticmethod
    def std(x, axis=0, keepdim=True):
        if isinstance(x, np.ndarray):
            return x.std(axis=axis, keepdims=keepdim)
        if isinstance(x, torch.Tensor):
            return x.std(dim=axis, unbiased=False, keepdim=keepdim)
        raise NotImplementedError('upsupport data type %s'%type(x))

    @staticmethod
    def cast(x, dtype='float'):
        if isinstance(x, np.ndarray):
            return x.astype(dtype)
        # if isinstance(x, tf.Tensor):
        #     return tf.cast(x, dtype)
        if isinstance(x, torch.Tensor):
            return x.type(getattr(torch, dtype))
        raise NotImplementedError('unsupported data type %s'%type(x))

    @staticmethod
    def maximum(x, v):
        if isinstance(x, np.ndarray):
            return np.maximum(x, v)
        # if isinstance(x, tf.Tensor):
        #     return tf.maximum(x, v)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min=v)
        raise NotImplementedError('unsupported data type %s'%type(x))
    
    @staticmethod
    def median(x, axis=0, keepdims=True):
        # NOTE: numpy will average when size is even,
        # but tensorflow and pytorch don't average
        if isinstance(x, np.ndarray):
            return np.median(x, axis=axis, keepdims=keepdims)
        # if isinstance(x, tf.Tensor):
        #     return tf.contrib.distributions.percentile(x, 50, axis=axis, keep_dims=keepdims)
        if isinstance(x, torch.Tensor):
            return torch.median(x, dim=axis, keepdim=keepdims)[0]
        raise NotImplementedError('unsupported data type %s'%type(x))
    
    @staticmethod
    def clip(x, min_val, max_val):
        if isinstance(x, np.ndarray):
            return np.clip(x, min_val, max_val)
        # if isinstance(x, tf.Tensor):
        #     return tf.clip_by_value(x, min_val, max_val)
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min_val, max_val)
        raise NotImplementedError('unsupported data type %s'%type(x))


def z_score(x, axis=0):
    mean = K.mean(x, axis=axis)
    std = K.std(x, axis=axis)
    return (x-mean) / (std + EPS)

def logcosh(pred, label, dim=0):
    loss = K.log(K.cosh(pred - label))
    return K.mean(loss, axis=dim)

def standard_mse(pred, label, axis=1):
    loss = (pred - label)**2
    return K.mean(loss, axis=axis, keepdim=False) 

def standard_rmse(pred, label, axis=1):
    return torch.sqrt(torch.mean(pred-label)**2)

def standard_mae(pred, label, axis=1):
    loss = torch.abs(pred - label)
    return K.mean(loss, axis=axis, keepdim=False)

def standard_cross_entropy(pred, label, reduce=True):
    y = K.cast(label > 0, 'float')
    p = pred # alias
    loss = K.maximum(p, 0) - p * y + K.log1p(K.exp(-K.abs(p)))
    if reduce:
        return K.mean(loss, keepdim=False)
    return loss


def batch_corr(x, y, axis=1, dim=0, keepdim=False):
    x = z_score(x, axis=axis)
    y = z_score(y, axis=axis)
    return K.mean(x*y, axis=axis, keepdim=keepdim)


def get_loss_fn(loss_fn):
    if loss_fn == 'mse':
        return standard_mse
    elif loss_fn == 'logcosh':
        return logcosh
    elif loss_fn == 'std_ce':
        return standard_cross_entropy
    else:
        raise NotImplementedError('loss function %s is not implemented'%loss_fn)

def get_metric_fn(eval_metric):
    # reflection: legacy name
    if eval_metric == 'corr':
        return batch_corr
    if eval_metric == 'mae':
        return standard_mae
    if eval_metric == 'rmse':
        return standard_rmse
    else:
        raise NotImplementedError('metric function %s is not implemented'%eval_metric)
    