"""Fingerprint-based observation featurization."""

import numpy as np
from rdkit.Chem import rdFingerprintGenerator

# Import-time constants: the Morgan generator + observation length need the
# fingerprint radius/length BEFORE any runtime config exists. These are NOT
# user-overridable; source them from the EnvCfg dataclass defaults.
from src.config import ENV_DEFAULTS

FINGERPRINT_LENGTH = ENV_DEFAULTS.fingerprint_length
FINGERPRINT_RADIUS = ENV_DEFAULTS.fingerprint_radius

morganFingerprintGen = rdFingerprintGenerator.GetMorganGenerator(fpSize=FINGERPRINT_LENGTH, radius=FINGERPRINT_RADIUS)


def get_observations(fp, remaining_steps):
    return np.append(np.array(fp, dtype='uint8'), remaining_steps)


def get_observations_from_list(fp_list, remaining_steps):
    a = np.zeros(FINGERPRINT_LENGTH + 1, dtype='uint8')
    for f in fp_list:
        a[f] = 1
    a[-1] = remaining_steps
    return a
