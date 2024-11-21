#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2024/5/25
# __author__: 'Alex Lu'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.fft import fft
import pywt

def identity_transform(data):
    return data


def fourier_transform(data):
    return fft(data)


def fd_transform(data):
    """ First Derivative. Row based transform.
    """
    return np.gradient(data, 1, axis=-1)


def sd_transform(data):
    """ Second Derivative. Row based transform.
    """
    # 计算每个样本的平均值
    return np.gradient(data, 2, axis=-1)


def range_scaling_transform(data):
    """
    range_scaling = min_max_scaling. Row based transform.
    Args:
        data ():

    Returns:
    """
    min_vals = data.min(axis=-1, keepdims=True)
    max_vals = data.max(axis=-1, keepdims=True)
    return (data - min_vals) / (max_vals - min_vals)


def snv_transform(data):
    """
    Standard Normal Variate. Row based transform.
    Why SNV? because all features are provided by one instrument at same time. SNV can reduce batch effects.
    """
    # 计算每个样本的平均值
    mean = np.mean(data, axis=-1, keepdims=True)
    # 计算每个样本的标准差
    std_dev = np.std(data, axis=-1, keepdims=True, ddof=1)
    # 对每个样本进行 SNV 变换
    snv_data = (data - mean) / std_dev
    return snv_data


def l2norm_transform(data):
    """ Row based transform. """
    # 计算每行的 L2 范数
    l2_norms = np.linalg.norm(data, axis=-1, keepdims=True)
    # 对每行除以该行的 L2 范数
    return data / l2_norms


def smooth_transform(data):
    """
    Row based transform.
    Alex: rampy lib can be used to get many kinds of smooth.
    e.g. savgol, whittaker, flat, hanning, hamming, bartlett, blackman etc.
    """
    window_length = 11
    poly_order = 3
    return np.apply_along_axis(savgol_filter, axis=-1, arr=data, window_length=window_length,
                               polyorder=poly_order)

def denoise_by_wavelets(data, wavelet= 'db4', level=4):
    wavelet = 'db4'
    level = 4
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs = [pywt.threshold(coeff, value=0.1, mode='soft') for coeff in coeffs]
    return pywt.waverec(coeffs, wavelet)

class Interp1dResize:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        input_size = data.shape[-1]
        old_coord_x = np.arange(0, input_size)
        new_coord_x = np.linspace(0, input_size - 1, self.output_size)
        interpolator = interp1d(old_coord_x, data, kind='cubic')
        return interpolator(new_coord_x)  # smoothing

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.output_size})"


class AdaptiveAvgPool1dResize:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        input_size = data.shape[-1]
        if input_size < self.output_size:   # padding 0 in both end
            lp = (self.output_size - input_size) // 2
            rp = self.output_size - input_size - lp
            data = np.pad(data, (lp, rp), mode='edge')
            return data

        data = torch.tensor(data, dtype=torch.float32)
        return F.adaptive_avg_pool1d(data, self.output_size).numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.output_size})"


class LambdaTransform:
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError(f"Argument lambd should be callable, got {repr(type(lambd).__name__)}")
        self.lambd = lambd

    def __call__(self, data):
        return self.lambd(data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SpectraDataset(Dataset):
    """
    Only support IR & Raman spectrum because continuous.
    Args:
        x (np.ndarray):  shape (num_data, spectrum_length), spectrum_length is width in output_shape.
        y (np.ndarray):
        output_shape (int | tuple(int, int) | tuple(int, int, int)):
            output one data's shape: (C, H, W) (channel, height, width).
            C, W must not be zero. Default C = 1.
            (1, 0, 1024) => height is 0 (no height). It is 1D data (in deep learning, there needs channel).
            3D shape is for modeles in timm lib which support img wiht shape CWH.
        num_copy (int): how many copies for dataset.
        transform (Callable): transform input x.
        generators (list(Callable)): generator function for generating data in channel.
        generate_mode (char): 'C' means generate data in Channel, 'H' means generate data in Height.
    """

    def __init__(self, x, y, output_shape=(1, 0, 1024), num_copy=1, transform=None, generators=None, generate_mode='C'):
        # ----- S (check & valid) ---------------------------------------------
        if isinstance(output_shape, int):
            output_shape = (1, 0, output_shape)
        elif isinstance(output_shape, tuple):
            if len(output_shape) == 2:
                output_shape = (output_shape[0], 0, output_shape[1])
            elif len(output_shape) > 3:
                raise RuntimeError(f"target shape ({output_shape}) only support 1d, 2d, 3d.")

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        # ----- E (check & valid) ---------------------------------------------

        if transform is not None:
            x = transform(x)

        self.num_copy = num_copy if num_copy is not None and num_copy > 0 else 1
        self.data_cache = [None] * len(x)  # (x, y)

        self.generators = [identity_transform] if generators is None or len(generators) == 0 else generators
        self.data_shape = output_shape

        if output_shape[0] == 1 and output_shape[1] > 0:
            generate_mode = 'H'
        target_num = self.data_shape[0] if 'C' == generate_mode.upper() else self.data_shape[1]

        for i in range(x.shape[0]):
            rows = self.__generate_data__(x[i], target_num)
            data = torch.tensor(np.array(rows), dtype=torch.float32)    # (target_num, W)
            if 'C' == generate_mode.upper():
                if output_shape[1] > 0:
                    repeat_h = output_shape[1]
                    # shape: [C, H, W]
                    data = data.reshape(output_shape[0], 1, -1).repeat(1, repeat_h, 1)
            elif 'H' == generate_mode.upper():
                repeat_c = output_shape[0]
                data = data.reshape(1, output_shape[1], -1).repeat(repeat_c, 1, 1)

            self.data_cache[i] = data, torch.tensor(y[i], dtype=torch.long)

    def __generate_data__(self, row, target_num):
        channels = [None] * target_num
        for j in range(target_num):
            i = j % len(self.generators)    # 循环填充
            channels[j] = self.generators[i](row)
        return channels

    def __len__(self):
        return len(self.data_cache) * self.num_copy

    def __getitem__(self, idx):
        seq = idx % len(self.data_cache)
        return self.data_cache[seq]
