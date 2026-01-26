
import sys
import torch
import pytorch_lightning.callbacks.model_checkpoint
from pytorch_lightning.utilities.upgrade_checkpoint import main as pl_upgrade_main
import os

# 1. Register Safe Globals
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint])

# 2. Monkeypatch torch.load to be permissive
# The PL upgrade utility calls torch.load internally without options we can control.
# We force it to trust the file.
original_load = torch.load

def permissive_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = permissive_load

def upgrade_file(path):
    print(f"Processing {path}...")
    # Inject arguments into sys.argv so argparse inside PL reads them
    sys.argv = ["upgrade_checkpoint", path]
    try:
        pl_upgrade_main()
        print(f" -> Success: {path}")
    except Exception as e:
        print(f" -> Failed: {path} with error {e}")

if __name__ == "__main__":
    base_dir = "refs/SynNet/tests/data/ref"
    ckpts = ["act.ckpt", "rt1.ckpt", "rxn.ckpt", "rt2.ckpt"]
    
    for ckpt in ckpts:
        full_path = os.path.join(base_dir, ckpt)
        if os.path.exists(full_path):
            upgrade_file(full_path)
        else:
            print(f"File not found: {full_path}")
