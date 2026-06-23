"""IP predictor orchestrator for the RL reward.

Owns the AIMNet-NSE ensemble (5 cross-validation models), the IP scaler, an
``ETKDGEmbedder`` for 3D conformer generation, and the IP reward factor. A *pure*
predictor — IP is NOT cached (its value depends on a random ETKDG conformer), so
this class has no cache and ``predict_IP`` runs every call. The BDE_IP reward
wraps it in ``cached(cache=None, call_on_empty=False)`` only to reuse the
generic dedup + index-mapping.
"""
import torch

from src.config import RewardCfg
from src.reward.ip.model import ev2kcal_per_mol, AimnetNseModel
from src.reward.scaler import get_scaler
from src.reward.ip.etkdg import ETKDGEmbedder

# All 5 cross-validation models; cv4 is the original single live model.
_IP_CV_PATHS = [
    'aimnetnse-models/aimnet-nse-cv0.jpt',
    'aimnetnse-models/aimnet-nse-cv1.jpt',
    'aimnetnse-models/aimnet-nse-cv2.jpt',
    'aimnetnse-models/aimnet-nse-cv3.jpt',
    'aimnetnse-models/aimnet-nse-cv4.jpt',
]
_IP_CV4_PATH = ['aimnetnse-models/aimnet-nse-cv4.jpt']


class IPPredictor:
    """Predicts ionization potentials via AIMNet-NSE (pure; no cache).

    ``ip_ensemble`` selects the model set: True -> average all 5 cv models;
    False -> the single cv4 model (the original live model). All valid molecules
    are run through ONE zero-padded forward (the ``.jpt`` model masks padded
    atoms via ``numbers==0``), so padding does not affect real-atom energies.
    """

    def __init__(self, device, etkdg_threads,
                 etkdg_max_attempts_cache, etkdg_max_attempts_uncache,
                 ip_factor=RewardCfg.ip_factor,
                 ip_ensemble=RewardCfg.ip_ensemble):
        self.device = device
        self.ip_factor = ip_factor
        self.ip_ensemble = bool(ip_ensemble)
        self.etkdg_max_attempts_cache = etkdg_max_attempts_cache
        self.etkdg_max_attempts_uncache = etkdg_max_attempts_uncache

        self.etkdg = ETKDGEmbedder(device, etkdg_threads)

        self.ip_scaler = get_scaler('./Data/anti-ip.csv')
        self.ip_model_path = _IP_CV_PATHS if self.ip_ensemble else _IP_CV4_PATH
        # ONE wrapper holding an EnsembleCalculator over the chosen path(s).
        self.ip_model = AimnetNseModel(self.ip_model_path, self.device)

    def predict_IP(self, molecules, maxAttempts):
        """Pure prediction. Returns ``(values, valids)`` parallel to ``molecules``.
        ``molecules`` may be empty.

        All valid molecules are batched through ONE zero-padded forward; each
        molecule contributes 3 contiguous rows (cation, neutral, anion) over the
        SAME conformer. ``ip = E_cation - E_neutral`` (eV), converted to kcal/mol.
        """
        embeds, vs = self.etkdg.embed(molecules, maxAttempts)

        valid_idx = [i for i, v in enumerate(vs) if v]
        if not valid_idx:
            return [0.0 for _ in molecules], vs

        n_valid = len(valid_idx)
        maxN = max(len(embeds[i][1]) for i in valid_idx)

        # Padded per-charge-state rows: molecule m occupies rows 3m, 3m+1, 3m+2.
        coord = torch.zeros((3 * n_valid, maxN, 3), dtype=torch.float)
        numbers = torch.zeros((3 * n_valid, maxN), dtype=torch.long)
        for m, i in enumerate(valid_idx):
            coords_np, numbers_list = embeds[i]
            natoms = len(numbers_list)
            c = torch.tensor(coords_np, dtype=torch.float)
            nums = torch.tensor(numbers_list, dtype=torch.long)
            # same conformer repeated across the 3 charge states
            coord[3 * m:3 * m + 3, :natoms, :] = c.unsqueeze(0)
            numbers[3 * m:3 * m + 3, :natoms] = nums.unsqueeze(0)

        charge = torch.tensor([1, 0, -1], dtype=torch.long).repeat(n_valid)   # cation, neutral, anion
        mult = torch.tensor([2, 1, 2], dtype=torch.long).repeat(n_valid)

        data = dict(
            coord=coord.to(self.device),
            numbers=numbers.to(self.device),
            charge=charge.to(self.device),
            mult=mult.to(self.device),
        )

        # disable optimizations for safety. with some combinations of pytorch/cuda it's getting very slow
        with torch.jit.optimized_execution(False), torch.no_grad():
            out = self.ip_model.model(data)

        e = out['energy'].reshape(n_valid, 3)
        ip_ev = e[:, 0] - e[:, 1]                     # E_cation - E_neutral
        ip_kcal = ev2kcal_per_mol(ip_ev).cpu().tolist()

        preds = [0.0 for _ in molecules]
        for m, i in enumerate(valid_idx):
            preds[i] = ip_kcal[m]
        return preds, vs
