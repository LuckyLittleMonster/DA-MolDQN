"""Fingerprint-based observation featurization."""

import numpy as np
from rdkit.Chem import rdFingerprintGenerator

from src import config_defaults as hyp


morganFingerprintGen = rdFingerprintGenerator.GetMorganGenerator(fpSize=hyp.fingerprint_length, radius=hyp.fingerprint_radius)


def get_observations(fp, remaining_steps):
    return np.append(np.array(fp, dtype='uint8'), remaining_steps)


def get_observations_from_list(fp_list, remaining_steps):
    a = np.zeros(hyp.fingerprint_length + 1, dtype='uint8')
    for f in fp_list:
        a[f] = 1
    a[-1] = remaining_steps
    return a
