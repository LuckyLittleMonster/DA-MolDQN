"""IP predictor orchestrator for the RL reward.

Owns the AIMNet-NSE ensemble (5 cross-validation models), the IP scaler, an
``ETKDGEmbedder`` for 3D conformer generation, and the IP reward factor. A *pure*
predictor — IP is NOT cached (its value depends on a random ETKDG conformer), so
this class has no cache and ``predict_IP`` runs every call. The BDE_IP reward
wraps it in ``cached(cache=None, call_on_empty=False)`` only to reuse the
generic dedup + index-mapping.
"""
import numpy as np
import torch

from src import config_defaults as hyp
from src.eval import to_numpy
from src.reward.ip import (ev2kcal_per_mol, calc_react_idx, get_scaler,
                           AimnetNseModel)
from src.reward.ip_predictor.etkdg import ETKDGEmbedder


class IPPredictor:
    """Predicts ionization potentials via an AIMNet-NSE ensemble (pure; no cache)."""

    def __init__(self, device, etkdg_threads,
                 etkdg_max_attempts_cache, etkdg_max_attempts_uncache):
        self.device = device
        self.ip_factor = hyp.ip_factor
        self.etkdg_max_attempts_cache = etkdg_max_attempts_cache
        self.etkdg_max_attempts_uncache = etkdg_max_attempts_uncache

        self.etkdg = ETKDGEmbedder(device, etkdg_threads)

        self.ip_scaler = get_scaler('./Data/anti-ip.csv')
        self.ip_model_path = [
            'aimnetnse-models/aimnet-nse-cv0.jpt',
            'aimnetnse-models/aimnet-nse-cv1.jpt',
            'aimnetnse-models/aimnet-nse-cv2.jpt',
            'aimnetnse-models/aimnet-nse-cv3.jpt',
            'aimnetnse-models/aimnet-nse-cv4.jpt']
        self.ip_model = [AimnetNseModel(ipmp, self.device) for ipmp in self.ip_model_path]

    def predict_IP(self, molecules, maxAttempts):
        """Pure prediction. Returns ``(values, valids)`` parallel to ``molecules``.
        ``molecules`` may be empty."""
        ds, vs = self.etkdg.rwmol2data_atts(molecules, maxAttempts)

        preds = []
        for data, valid in zip(ds, vs):
            if valid:
                model_id = np.random.randint(0, len(self.ip_model))
                model_id = 4  # todo : use random model
                ip_model = self.ip_model[model_id]
                # disable optimizations for safety. with some combinations of pytorch/cuda it's getting very slow
                with torch.jit.optimized_execution(False), torch.no_grad():
                    pred = ip_model.model(data)
                pred['charges'] = pred['charges'].sum(-1)
                pred = to_numpy(pred)
                # calculate indicies
                pred.update(calc_react_idx(pred))
                # write
                for k, v in pred.items():
                    pred[k] = v.tolist()
                pred_ip = ev2kcal_per_mol(pred['ip'])
                preds.append(pred_ip)
            else:
                preds.append(0.0)

        return preds, vs
