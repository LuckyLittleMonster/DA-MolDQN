"""Fast ADMET-AI wrapper bypassing Lightning Trainer overhead.

Speedups vs ADMETModel:
- Direct model.forward() instead of pl.Trainer.predict()
- Pre-sorted DrugBank arrays for O(log n) percentiles via np.searchsorted
- LRU cache for MolGraph featurization (saves re-featurizing repeated SMILES)
- No tqdm for small batches
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from chemprop.data.collate import BatchMolGraph
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import load_model
from chemprop.models.utils import load_output_columns

from admet_ai.constants import DEFAULT_DRUGBANK_PATH, DEFAULT_MODELS_DIR
from admet_ai.drugbank import filter_drugbank_by_atc, read_drugbank_data
from admet_ai.physchem import PHYSCHEM_PROPERTY_TO_FUNCTION
from admet_ai.utils import get_drugbank_suffix


class FastADMETModel:
    """Fast ADMET-AI wrapper bypassing Lightning Trainer overhead.

    Drop-in replacement for ``ADMETModel`` with identical output format.

    Parameters
    ----------
    include_physchem : bool
        Whether to include physicochemical properties in predictions.
    drugbank_percentiles : bool
        Whether to compute DrugBank percentile ranks. Set False for RL speed.
    cache_size : int
        Max number of SMILES->MolGraph entries to cache. 0 disables caching.
    device : str
        Torch device for model inference ('cpu' or 'cuda').
    """

    def __init__(
        self,
        include_physchem: bool = True,
        drugbank_percentiles: bool = True,
        cache_size: int = 4096,
        device: str | None = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.include_physchem = include_physchem
        self.drugbank_percentiles = drugbank_percentiles

        # Load featurizer (same default as chemprop MoleculeDataset)
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

        # Load all 10 MPNN models (2 ensembles x 5 models)
        self.task_lists: list[list[str]] = []
        self.model_lists: list[list[torch.nn.Module]] = []
        self._load_models(DEFAULT_MODELS_DIR)

        # Pre-sort DrugBank columns for O(log n) percentile lookup
        self._drugbank_sorted: dict[str, np.ndarray] = {}
        self._drugbank_suffix: str = ""
        if drugbank_percentiles:
            self._prepare_drugbank(DEFAULT_DRUGBANK_PATH)

        # Set up LRU cache for SMILES -> MolGraph
        if cache_size > 0:
            self._featurize_cached = lru_cache(maxsize=cache_size)(self._featurize_smiles)
        else:
            self._featurize_cached = self._featurize_smiles

    @staticmethod
    def _fix_torchmetrics_device(model):
        """Fix torchmetrics Metric modules that may have stale device refs (e.g. MPS)."""
        try:
            from torchmetrics import Metric
        except ImportError:
            return
        for module in model.modules():
            if isinstance(module, Metric):
                # Reset internal device to CPU so .to() can work
                module._device = torch.device("cpu")

    def _load_models(self, models_dir: Path | str) -> None:
        """Load model ensembles and move to device in eval mode."""
        model_dirs = sorted(Path(models_dir).iterdir())
        for model_dir in model_dirs:
            model_paths = sorted(model_dir.glob("**/*.pt"))
            task_names = load_output_columns(model_paths[0])
            models = []
            for mp in model_paths:
                model = load_model(mp, multicomponent=False)
                model.eval()
                self._fix_torchmetrics_device(model)
                model.to(self.device)
                models.append(model)
            self.task_lists.append(task_names)
            self.model_lists.append(models)

    def _prepare_drugbank(self, drugbank_path: Path) -> None:
        """Pre-sort DrugBank reference columns for fast searchsorted percentiles."""
        drugbank = read_drugbank_data(drugbank_path)
        drugbank_filtered = filter_drugbank_by_atc(atc_code=None, drugbank=drugbank)
        self._drugbank_suffix = get_drugbank_suffix(atc_code=None)

        # Pre-sort each numeric column (drop NaN, sort ascending)
        for col in drugbank_filtered.columns:
            series = drugbank_filtered[col]
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna().values.astype(np.float64)
                if len(clean) > 0:
                    self._drugbank_sorted[col] = np.sort(clean)

    def _featurize_smiles(self, smiles: str) -> MolGraph | None:
        """Featurize a single SMILES string into a MolGraph."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return self.featurizer(mol)

    def predict(self, smiles: str | list[str]) -> dict[str, float] | pd.DataFrame:
        """Predict ADMET properties for one or more SMILES.

        Drop-in replacement for ``ADMETModel.predict()``.

        Parameters
        ----------
        smiles : str or list[str]
            SMILES string(s) to predict.

        Returns
        -------
        dict[str, float] or pd.DataFrame
            Single SMILES -> dict; list -> DataFrame with SMILES index.
        """
        single = isinstance(smiles, str)
        if single:
            smiles = [smiles]

        # Featurize and filter invalid
        mgs: list[MolGraph] = []
        valid_smiles: list[str] = []
        valid_mols: list[Chem.Mol] = []
        for smi in smiles:
            mg = self._featurize_cached(smi)
            if mg is not None:
                mgs.append(mg)
                valid_smiles.append(smi)
                if self.include_physchem:
                    valid_mols.append(Chem.MolFromSmiles(smi))

        if len(valid_smiles) < len(smiles):
            print(f"Warning: {len(smiles) - len(valid_smiles):,} invalid molecules removed.")

        if not mgs:
            return {} if single else pd.DataFrame()

        # Build BatchMolGraph and run inference
        bmg = BatchMolGraph(mgs)
        bmg.to(self.device)
        task_to_preds = self._forward_ensembles(bmg)

        # Build DataFrame
        admet_preds = pd.DataFrame(task_to_preds, index=valid_smiles)

        # Physchem properties (no tqdm)
        if self.include_physchem:
            physchem_data = [
                {name: fn(mol) for name, fn in PHYSCHEM_PROPERTY_TO_FUNCTION.items()}
                for mol in valid_mols
            ]
            physchem_preds = pd.DataFrame(physchem_data, index=valid_smiles)
            preds = pd.concat((physchem_preds, admet_preds), axis=1)
        else:
            preds = admet_preds

        # DrugBank percentiles (vectorized searchsorted)
        if self.drugbank_percentiles and self._drugbank_sorted:
            preds = self._add_percentiles(preds, valid_smiles)

        if single:
            return preds.iloc[0].to_dict()
        return preds

    def predict_properties(self, smiles: str | list[str]) -> dict[str, float] | pd.DataFrame:
        """Predict only ADMET model properties (no physchem, no percentiles).

        Fastest path for RL reward computation.
        """
        single = isinstance(smiles, str)
        if single:
            smiles = [smiles]

        mgs: list[MolGraph] = []
        valid_smiles: list[str] = []
        for smi in smiles:
            mg = self._featurize_cached(smi)
            if mg is not None:
                mgs.append(mg)
                valid_smiles.append(smi)

        if not mgs:
            return {} if single else pd.DataFrame()

        bmg = BatchMolGraph(mgs)
        bmg.to(self.device)
        task_to_preds = self._forward_ensembles(bmg)

        preds = pd.DataFrame(task_to_preds, index=valid_smiles)
        if single:
            return preds.iloc[0].to_dict()
        return preds

    def _forward_ensembles(self, bmg: BatchMolGraph) -> dict[str, np.ndarray]:
        """Run forward pass through all model ensembles.

        Bypasses Lightning Trainer entirely -- direct model(bmg) calls.
        """
        task_to_preds: dict[str, np.ndarray] = {}

        with torch.inference_mode():
            for tasks, models in zip(self.task_lists, self.model_lists):
                # Stack predictions from all models in the ensemble
                ensemble_preds = torch.stack([model(bmg) for model in models], dim=0)
                # Average across ensemble members: (n_models, batch, n_tasks) -> (batch, n_tasks)
                avg_preds = ensemble_preds.mean(dim=0).cpu().numpy()

                for i, task in enumerate(tasks):
                    task_to_preds[task] = avg_preds[:, i]

        return task_to_preds

    def _add_percentiles(self, preds: pd.DataFrame, smiles: list[str]) -> pd.DataFrame:
        """Add DrugBank percentile columns using vectorized searchsorted.

        Matches scipy.stats.percentileofscore(kind='rank') behavior:
            percentile = (count_below + 0.5 * count_equal) / total * 100
        which equals: 0.5 * (searchsorted_left + searchsorted_right) / total * 100
        """
        percentile_data: dict[str, np.ndarray] = {}

        for col in preds.columns:
            sorted_arr = self._drugbank_sorted.get(col)
            if sorted_arr is not None:
                values = preds[col].values.astype(np.float64)
                left = np.searchsorted(sorted_arr, values, side="left")
                right = np.searchsorted(sorted_arr, values, side="right")
                percentile_data[f"{col}_{self._drugbank_suffix}"] = (
                    0.5 * (left + right) / len(sorted_arr) * 100
                )

        percentile_df = pd.DataFrame(percentile_data, index=smiles)
        return pd.concat([preds, percentile_df], axis=1)

    def clear_cache(self) -> None:
        """Clear the MolGraph LRU cache."""
        if hasattr(self._featurize_cached, "cache_clear"):
            self._featurize_cached.cache_clear()

    def cache_info(self):
        """Return LRU cache statistics."""
        if hasattr(self._featurize_cached, "cache_info"):
            return self._featurize_cached.cache_info()
        return None
