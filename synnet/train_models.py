
"""
Train the 4 SynNet networks: Action, Reactant1, Reaction, Reactant2.
"""
import argparse
import logging
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from scipy import sparse

from synnet.models.mlp import MLP, load_array

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing Xy .npz files")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--step", type=str, choices=["act", "rt1", "rxn", "rt2", "all"], default="all")
    parser.add_argument("--features", type=str, default="fp", help="Feature type (fp)")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10) # 10 epochs for faster training in this phase
    parser.add_argument("--ncpu", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()

def train_network(args, step, input_dim, output_dim, task, loss, valid_loss="mse"):
    print(f"\n=== Training {step} Network ===")
    
    # Paths
    ref_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir) / step
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    print(f"Loading data from {ref_dir}...")
    try:
        X_train = sparse.load_npz(ref_dir / f"X_{step}_train.npz")
        y_train = sparse.load_npz(ref_dir / f"y_{step}_train.npz")
        X_valid = sparse.load_npz(ref_dir / f"X_{step}_valid.npz")
        y_valid = sparse.load_npz(ref_dir / f"y_{step}_valid.npz")
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        return

    # Convert to Tensor
    X_train = torch.Tensor(X_train.toarray())
    y_train_tensor = torch.Tensor(y_train.toarray()) if task == "regression" else torch.LongTensor(y_train.toarray().reshape(-1))
    
    X_valid = torch.Tensor(X_valid.toarray())
    y_valid_tensor = torch.Tensor(y_valid.toarray()) if task == "regression" else torch.LongTensor(y_valid.toarray().reshape(-1))

    print(f"Train shapes: X={X_train.shape}, y={y_train_tensor.shape}")
    
    # Data Loaders
    train_iter = load_array((X_train, y_train_tensor), args.batch_size, ncpu=args.ncpu, is_train=True)
    valid_iter = load_array((X_valid, y_valid_tensor), args.batch_size, ncpu=args.ncpu, is_train=False)

    # Model
    # Using small architecture for this phase (100 hidden, 3 layers) which matches test models.
    # Can be scaled up for "Production" later by changing args or defaults.
    mlp = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=100, 
        num_layers=3,   
        dropout=0.5,
        num_dropout_layers=1,
        task=task,
        loss=loss,
        valid_loss=valid_loss,
        optimizer="adam",
        learning_rate=args.lr,
        ncpu=args.ncpu,
    )

    # Trainer
    tb_logger = pl_loggers.TensorBoardLogger(str(save_dir / "logs"))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(save_dir),
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        accelerator="auto", 
        devices=1 if torch.cuda.is_available() else None
    )

    print("Starting training...")
    trainer.fit(mlp, train_iter, valid_iter)
    
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
    
    # Also save as 'ckpts.dummy-val_loss=0.00.ckpt' to match SynthEnv expectations
    # if we point SynthEnv to this dir later.
    final_path = save_dir.parent / f"{step}.ckpt"
    print(f"Copying best model to {final_path} for easy loading...")
    shutil.copy(checkpoint_callback.best_model_path, final_path)


def main():
    args = get_args()
    
    # Dimensions (Based on 4096-bit fingerprints and 256-bit embeddings)
    nbits = args.nbits
    emb_dim = 256 # For Rt1, Rt2 output
    num_rxns = 91 # HB templates
    
    if args.step in ["act", "all"]:
        # Action: Input = 3 * nbits (Target + Mol1 + Mol2/None) -> Output 4 (Add, Expand, Merge, End)
        train_network(args, "act", 3 * nbits, 4, "classification", "cross_entropy", "accuracy")

    if args.step in ["rt1", "all"]:
        # Rt1: Input = 3 * nbits -> Output 256 (Embedding)
        train_network(args, "rt1", 3 * nbits, emb_dim, "regression", "mse", "mse")

    if args.step in ["rxn", "all"]:
        # Rxn: Input = 4 * nbits (Target + Mol1 + Mol2 + Rt1_Emb?? No wait)
        # Check prep_utils.py:
        # X_rxn = hstack([states, steps[:, (2*out_dim+2):]]) -> State(3*4096) + Mol1_FP(4096) = 4*4096
        # Output = num_rxns (91)
        train_network(args, "rxn", 4 * nbits, num_rxns, "classification", "cross_entropy", "accuracy")

    if args.step in ["rt2", "all"]:
        # Rt2: Input = Output of Prep: 3*4096 + 4096 + 91 = 4*nbits + num_rxns
        # Output = 256 (Embedding)
        train_network(args, "rt2", 4 * nbits + num_rxns, emb_dim, "regression", "mse", "mse")

if __name__ == "__main__":
    main()
