
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks.model_checkpoint
import os
import sys

# Register safe globals to allow unpickling legacy checkpoints
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint])

def upgrade_checkpoint(path):
    print(f"Upgrading {path}...")
    try:
        # Load with weights_only=False since we trust these local files
        # and we have registered the necessary globals foundation
        checkpoint = torch.load(path, weights_only=False)
        
        # Re-save effectively upgrades it to the current PL version format if changed
        # usage of 'pytorch_lightning.utilities.migration.pl_legacy_patch' might be needed internally 
        # but usually just re-saving with new library version does the trick for simple dict incompatibility
        torch.save(checkpoint, path)
        print(f"Successfully upgraded {path}")
    except Exception as e:
        print(f"Failed to upgrade {path}: {e}")

if __name__ == "__main__":
    base_dir = "refs/SynNet/tests/data/ref"
    ckpts = ["act.ckpt", "rt1.ckpt", "rxn.ckpt", "rt2.ckpt"]
    
    for ckpt in ckpts:
        full_path = os.path.join(base_dir, ckpt)
        if os.path.exists(full_path):
            upgrade_checkpoint(full_path)
        else:
            print(f"File not found: {full_path}")
