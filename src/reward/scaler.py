"""Shared MinMax scaler loader for the BDE and IP reward components.

Both ``bde/predictor.py`` and ``ip/predictor.py`` fit a MinMaxScaler from a
reference CSV (``./Data/anti-bde.csv`` / ``./Data/anti-ip.csv``). The cached
fast-path hardcodes the precomputed min/max for those two files so the scaler is
identical without re-reading the CSV.
"""
import csv

import numpy as np
from sklearn import preprocessing


def _get_scaler(path, real_col_id=1):
    real = []
    with open(path) as f:
        s = csv.reader(f, delimiter="\t")
        next(s)
        for r in s:
            if r[real_col_id] != '':
                real.append([float(r[real_col_id])])
    return preprocessing.MinMaxScaler().fit(real)


def get_scaler(path, real_col_id=1, use_cache=True):
    if use_cache:
        if 'bde' in path:
            # ./Data/anti-bde.csv -> (482, 1), max 96.586..., min 59.795...
            data = np.array([[96.58618528], [59.79533261]])
            return preprocessing.MinMaxScaler().fit(data)
        elif 'ip' in path:
            # ./Data/anti-ip.csv -> (445, 1), max 178.162..., min 110.830...
            data = np.array([[178.1623553], [110.8306396]])
            return preprocessing.MinMaxScaler().fit(data)
    return _get_scaler(path, real_col_id)
