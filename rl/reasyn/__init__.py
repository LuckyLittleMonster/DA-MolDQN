"""ReaSyn molecular generation and RL modules.

Registers sys.modules aliases so pickles saved under the old top-level
'reasyn' package path can still be unpickled after the move to 'rl.reasyn'.
"""

import importlib
import sys

# Only alias the modules actually referenced inside pickle files
# (fpindex.pkl, matrix.pkl contain reasyn.chem.{fpindex,mol,matrix,reaction,stack})
_PICKLE_MODULES = (
    'chem', 'chem.fpindex', 'chem.mol', 'chem.matrix',
    'chem.reaction', 'chem.stack',
)

# sklearn compat: fpindex.pkl was pickled with older sklearn where ManhattanDistance
# existed; in sklearn >=1.8 it was renamed to ManhattanDistance64.
try:
    import sklearn.metrics._dist_metrics as _dm
    if not hasattr(_dm, 'ManhattanDistance'):
        _dm.ManhattanDistance = _dm.ManhattanDistance64
except Exception:
    pass

sys.modules.setdefault('reasyn', sys.modules[__name__])
for _sub in _PICKLE_MODULES:
    _old = f'reasyn.{_sub}'
    if _old not in sys.modules:
        try:
            sys.modules[_old] = importlib.import_module(f'rl.reasyn.{_sub}')
        except ImportError:
            pass
