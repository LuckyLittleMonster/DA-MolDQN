"""IP (ionization potential) reward component.

Houses the AIMNet-NSE based IP machinery extracted from agent.py:
reactivity-index features, the MinMax scaler loader, and the picklable
``AimnetNseModel`` wrapper around the ensemble calculator.
"""
import csv

import numpy as np
from sklearn import preprocessing

from src.eval import load_models


def ev2kcal_per_mol(ev):
    return ev * 23.0609


def calc_react_idx(data):
    ip = data['energy'][0] - data['energy'][1]
    ea = data['energy'][1] - data['energy'][2]
    f_el = data['charges'][1] - data['charges'][0]
    f_nuc = data['charges'][2] - data['charges'][1]
    chi = 0.5 * (ip + ea)
    eta = 0.5 * (ip - ea)
    omega = (chi ** 2) / (2 * eta)
    f_rad = 0.5 * (f_el + f_nuc)
    _omega = np.expand_dims(omega, axis=-1)
    omega_el = f_el * _omega
    omega_nuc = f_nuc * _omega
    omega_rad = f_rad * _omega
    return dict(ip=ip, ea=ea, f_el=f_el, f_nuc=f_nuc, f_rad=f_rad,
                chi=chi, eta=eta, omega=omega,
                omega_el=omega_el, omega_nuc=omega_nuc, omega_rad=omega_rad)


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


class AimnetNseModel():
    """The original model is not picklable and can't be used with spawn.
    This class is a wrapper of EnsembleCalculator."""

    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.model = load_models([path]).to(device)

    def __setstate__(self, state):
        self.path = state['path']
        self.device = state['device']
        self.model = load_models([self.path]).to(self.device)

    def __getstate__(self):
        return dict(path=self.path, device=self.device)
