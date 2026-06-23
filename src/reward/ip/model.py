"""IP (ionization potential) model + reactivity-index features.

Houses the AIMNet-NSE based IP machinery extracted from agent.py:
reactivity-index features, the eV->kcal/mol conversion, and the picklable
``AimnetNseModel`` wrapper around the ensemble calculator.
"""
import numpy as np

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


class AimnetNseModel():
    """The original model is not picklable and can't be used with spawn.
    This class is a wrapper of EnsembleCalculator.

    ``paths`` is a list of ``.jpt`` files: ``load_models`` builds one
    ``EnsembleCalculator`` over them (a single path -> just that model; multiple
    -> the average). One ``data`` dict is run through all models in one forward.
    """

    def __init__(self, paths, device):
        self.paths = list(paths)
        self.device = device
        self.model = load_models(self.paths).to(device)

    def __setstate__(self, state):
        self.paths = list(state['paths'])
        self.device = state['device']
        self.model = load_models(self.paths).to(self.device)

    def __getstate__(self):
        return dict(paths=self.paths, device=self.device)
