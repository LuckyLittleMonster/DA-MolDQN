"""Memoize a pure predictor by key, optionally backed by a swappable Cache."""


class CachedPredictor:
    """Wrap a pure predictor with de-duplication + index-mapping and an optional cache.

    ``predict_fn(keys, mols) -> (values, valids)`` is the pure predictor: it
    receives the de-duplicated, not-yet-cached keys (+ their molecules) and
    returns two parallel lists.

    ``cache`` is any :class:`~src.cache.base.Cache`, or ``None`` to disable
    caching — e.g. the IP predictor, whose value depends on a random conformer
    and must never be cached. Only ``valid`` results are cached.

    ``call_on_empty`` forces ``predict_fn`` to run even with zero uncached keys,
    for predictors whose call is itself a side effect that must happen every step.

    ``invalid_value`` fills the result slots of molecules that were dropped or
    failed prediction.
    """

    def __init__(self, predict_fn, cache=None, call_on_empty=False, invalid_value=None):
        self.predict_fn = predict_fn
        self.cache = cache
        self.call_on_empty = call_on_empty
        self.invalid_value = invalid_value

    def predict(self, smiles, smiles_p, use_cache=True):
        """``smiles``: per-molecule keys (with duplicates). ``smiles_p``: unique
        key -> ``(mol, [indices])``. Returns ``(values, valids)`` parallel to
        ``smiles``."""
        n = len(smiles)
        values = [self.invalid_value] * n
        valids = [False] * n
        do_cache = use_cache and self.cache is not None

        uncached_keys, uncached_mols = [], []
        for key, (mol, ids) in smiles_p.items():
            if do_cache:
                value, hit = self.cache.get(key)
                if hit:
                    for i in ids:
                        values[i] = value
                        valids[i] = True
                    continue
            uncached_keys.append(key)
            uncached_mols.append(mol)

        if uncached_keys or self.call_on_empty:
            preds, pred_valids = self.predict_fn(uncached_keys, uncached_mols)
            for key, value, valid in zip(uncached_keys, preds, pred_valids):
                if valid:
                    if do_cache:
                        self.cache.put(key, value)
                    for i in smiles_p[key][1]:
                        values[i] = value
                        valids[i] = True
        return values, valids
