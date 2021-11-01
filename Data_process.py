#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:54:10 2021

@author: andyq
"""

import wfdb
import numpy as np
from scipy import signal
import os
import pywt


def WTfilt_1d(sig):
    """
    Wavelet denoise
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt


def get_filename(path, file_type):
    get_filename = []
    filelist = os.listdir(path)
    for item in filelist:
        if item.endswith(file_type):
            get_filename.append(os.path.splitext(os.path.basename(item))[0])
    return get_filename


def varify_label(temp_symbols):
    """
    Parameters
    ----------
    temp_symbols : list
        temp_symbols contains the label of the segment that is for every heart beats of the segment.

    Returns
    -------
    varified_label : str

    """
    label = []
    N_class = ['N', 'L', 'R', 'e', 'j']
    for i in temp_symbols:
        if i not in N_class:
            label.append(i)

    if len(temp_symbols) == 0:
        varified_label = 'Q'

    elif len(label) == 0:
        varified_label = 'N'
    else:
        varified_label = max(label, key=label.count)

    if varified_label == '~' or varified_label == 's' or varified_label == 'T':
        varified_label = 'N'

    if (
        varified_label == 'A'
        or varified_label == 'a'
        or varified_label == 'J'
        or varified_label == 'S'
    ):
        varified_label = 'S'

    if varified_label == 'V' or varified_label == 'E':
        varified_label = 'V'

    if varified_label == 'F':
        varified_label = 'F'

    if (
        varified_label == '/'
        or varified_label == 'f'
        or varified_label == 'Q'
        or varified_label == 'P'
    ):
        varified_label = 'Q'

    return varified_label


def get_label(whole_signal, index, symbol, start_point, f, seg_length, step):
    """

    Parameters
    ----------
    whole_signal : list
        one piece of ecg signal
    index : list
        the list of the index to locate the label
    symbol : list
        the list of the symbol
    start_point : int
        the start point of the sliding window
    f : int
        the sampling frequency of the original data (Hz)
    seg_length : int
        the length of the segment
    step : int
        the distance of the sliding window moving forword

    Returns
    -------
    SEG_data : list
               the signal duration data
    SEG_label : list
                the label of the segment

    """
    SEG_label = []
    width_win = f * seg_length
    while start_point + width_win <= len(whole_signal):
        end_point = start_point + width_win
        # get the symbols
        temp_index = np.where((index <= end_point) & (index >= start_point))
        if len(temp_index[0]) == 0:
            temp_symbols = []
        else:
            first_point = temp_index[0][0]
            last_point = temp_index[0][-1]
            temp_symbols = symbol[first_point : last_point + 1]

        temp_label = varify_label(temp_symbols)

        SEG_label.append(temp_label)

        start_point = start_point + step
    return SEG_label


def get_SEG(
    whole_signal,
    index,
    symbol,
    start_point,
    f,
    seg_length,
    step,
    step_list,
    resample=False,
    resample_size=1250,
):
    """
    Parameters
    ----------
    whole_signal : array
        one vector of the signal
    index : list
        the list of the index to locate the label
    symbol : list
        the list of the symbol
    start_point : int
        the start point of the sliding window
    f : int
        the sampling frequency of the original data (Hz)
    seg_length : int
        the length of the segment
    step : int
        the distance of the sliding window moving forword
    step list : array
        the moving step of the sliding window
    resample : boolean,
        if true, resample the data with the new frequency. The default is False.
    resmaple size : int
    Returns
    -------
    SEG_data : list
               the signal duration data
    SEG_label : list
                the label of the segment

    """
    SEG_data = []
    SEG_label = []
    width_win = f * seg_length
    step_change = step
    while start_point + width_win <= len(whole_signal):
        end_point = start_point + width_win
        # slice the data
        temp_seq = whole_signal[start_point:end_point]
        # temp_seq = WTfilt_1d(temp_seq)
        # if the smapling frequency is not
        if resample is True:
            temp_seq = signal.resample(temp_seq, 1250)
        # get the symbols
        temp_index = np.where((index <= end_point) & (index >= start_point))
        if len(temp_index[0]) == 0:
            temp_symbols = []
        else:
            first_point = temp_index[0][0]
            last_point = temp_index[0][-1]
            temp_symbols = symbol[first_point : last_point + 1]

        temp_label = varify_label(temp_symbols)
        if temp_label == 'N':
            step_change = step  #####
        elif temp_label == 'S':
            step_change = step_list[0]
        elif temp_label == 'V':
            step_change = step_list[1]
        elif temp_label == 'F':
            step_change = step_list[2]
        elif temp_label == 'Q':
            step_change = step_list[3]

        SEG_label.append(temp_label)
        SEG_data.append(temp_seq)
        start_point = start_point + step_change
    return np.array(SEG_data), np.array(SEG_label)


def count_label(SEG_label):
    """
    Parameters
    ----------
    SEG_label : list
        label list.

    Returns
    -------
    count_list : array

    """
    N, S, V, F, Q = 0, 0, 0, 0, 0
    for i in range(len(SEG_label)):
        count_N = SEG_label[i].count('N')
        count_S = SEG_label[i].count('S')
        count_V = SEG_label[i].count('V')
        count_F = SEG_label[i].count('F')
        count_Q = SEG_label[i].count('Q')

        N += count_N
        S += count_S
        V += count_V
        F += count_F
        Q += count_Q
    count_list = [N, S, V, F, Q]
    return count_list


def cal_func(step, n, N, per):
    """

    Parameters
    ----------
    L : int
        the length of the segment that will be got.
    n : int
        the number of the segment type that will be balanced.
    N : int
        the most type in the five types.
    per : int(less than 1)
        the percentage of the most type which is used retaining
        in the final dataset
    Returns
    -------
    moving_stepving :int

    """
    N_res = N * per
    moving_step = round(step * (n / N_res))  # moving_step-150  is good for NST
    return moving_step


def cal_moving_step(path, start_point, f, seg_length, step, per):
    """
    Parameters
    ----------
    path : str
        the path of the database
    start_point : int
        the start point of the sliding window
    f : int
        the sampling frequency of the database
    seg_length : int
        the fixed duration of the segment
    step : int
        the basic moving forward size of the sliding window
    per : int(less than 1)
        the percentage of the type with the most data which is retained
        in the final dataset

    Returns
    -------
    step_list : array
        an arrary which shows the moving step

    """
    first_label = []
    filename_atr = get_filename(path, '.atr')
    filename_hea = get_filename(path, '.hea')
    for j in filename_atr:
        if j not in filename_hea:
            filename_atr.remove(j)
    for i in range(len(filename_atr)):
        annotation = wfdb.rdann(path + filename_atr[i], 'atr')
        # record_name = annotation.record_name
        index = annotation.sample
        symbol = np.array(annotation.symbol)
        # N_Seg, S_Seg, F_Seg, V_Seg, Q_Seg = hearbeats_seg(path)
        record = wfdb.rdrecord(path + filename_atr[i])
        signal_name = record.sig_name
        if 'V1' in signal_name:
            lead_index = record.sig_name.index('V1')
        elif 'V2' in signal_name:
            lead_index = record.sig_name.index('V2')
        elif 'V3' in signal_name:
            lead_index = record.sig_name.index('V3')
        elif 'V4' in signal_name:
            lead_index = record.sig_name.index('V4')
        elif 'V5' in signal_name:
            lead_index = record.sig_name.index('V5')
        elif 'ECG' in signal_name:
            lead_index = record.sig_name.index('ECG')
        sig = record.p_signal.transpose()
        whole_signal = sig[lead_index]
        if '+' in symbol:
            where_plus = np.where(symbol == '+')
            symbol = np.delete(symbol, where_plus[0])
            index = np.delete(index, where_plus[0])

        label = get_label(whole_signal, index, symbol, start_point, f, seg_length, step)
        first_label.append(label)
    count_list = count_label(first_label)
    max_num = np.max(count_list)
    # N_step = cal_func(step, count_list[0], max_num,per)
    S_step = cal_func(step, count_list[1], max_num, per)
    V_step = cal_func(step, count_list[2], max_num, per)
    F_step = cal_func(step, count_list[3], max_num, per)
    Q_step = cal_func(step, count_list[4], max_num, per)
    # step_list = [S_step, V_step, F_step, Q_step]
    step_list = [S_step, V_step, F_step, Q_step]
    return step_list


def get_dataset(
    path, seg_length, start_point, f, step, per, resample, resample_size=None
):
    """

    Parameters
    ----------
    path : str
        the path of the database.
    seg_length : int
        the fixed duration of the segment
    start_point : int
        the start point of the sliding window
    f : int
        Sampling frequency of database
    step :int
        the size of the sliding window m
    per : int(less than 1 or equal to 1)
        the percentage of the most type which is used to retain
        in the final dataset
    resample : boolean
        if true, resample the data with the new frequency. The default is False.
    resample_size : int

    Returns
    -------
    None.

    """
    # initialize the SEG_data and SEG_label
    if resample:
        segment_points = resample_size
    else:
        segment_points = f * seg_length
    SEG_data = np.empty((0, segment_points))
    SEG_label = np.empty((0))
    print("Loading database...")
    filename_atr = get_filename(path, '.atr')
    filename_hea = get_filename(path, '.hea')
    for j in filename_atr:
        if j not in filename_hea:
            filename_atr.remove(j)
    print("Calculating size of moving farward step...")
    step_list = cal_moving_step(path, start_point, f, seg_length, step, per)
    print("Getting Segments and Construct dataset...")
    for i in range(len(filename_atr)):
        annotation = wfdb.rdann(path + filename_atr[i], 'atr')
        record_name = annotation.record_name
        index = annotation.sample
        symbol = np.array(annotation.symbol)
        # N_Seg, S_Seg, F_Seg, V_Seg, Q_Seg = hearbeats_seg(path)
        record = wfdb.rdrecord(path + record_name)
        signal_name = record.sig_name
        if 'V1' in signal_name:
            lead_index = record.sig_name.index('V1')
        elif 'V2' in signal_name:
            lead_index = record.sig_name.index('V2')
        elif 'V3' in signal_name:
            lead_index = record.sig_name.index('V3')
        elif 'V4' in signal_name:
            lead_index = record.sig_name.index('V4')
        elif 'V5' in signal_name:
            lead_index = record.sig_name.index('V5')
        elif 'ECG' in signal_name:
            lead_index = record.sig_name.index('ECG')
        sig = record.p_signal.transpose()
        whole_signal = sig[lead_index]

        if '+' in symbol:
            where_plus = np.where(symbol == '+')
            symbol = np.delete(symbol, where_plus[0])
            index = np.delete(index, where_plus[0])
        data_1, label_1 = get_SEG(
            whole_signal,
            index,
            symbol,
            start_point,
            f,
            seg_length,
            step,
            step_list,
            resample,
            resample_size,
        )

        SEG_data = np.concatenate((SEG_data, data_1))
        SEG_label = np.concatenate((SEG_label, label_1))
    return SEG_data, SEG_label


path = '/Users/andyq/Downloads/ECG_Data/mit-bih-arrhythmia-database-1.0.0/'

SEG_data, SEG_label = get_dataset(
    path,
    seg_length=5,
    start_point=0,
    f=360,
    step=750,
    per=1,
    resample=True,
    resample_size=1250,
)

print(np.where(SEG_label == 'N')[0].shape[0])
print(np.where(SEG_label == 'S')[0].shape[0])
print(np.where(SEG_label == 'V')[0].shape[0])
print(np.where(SEG_label == 'F')[0].shape[0])
print(np.where(SEG_label == 'Q')[0].shape[0])
