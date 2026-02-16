#!/usr/bin/env python
"""
Prepare USPTO-FULL dataset for V3 link predictor training.

Converts USPTO_FULL.csv format:
  PatentNumber,Year,reactions
  US03930836,1976,CCOCC.CCS(=O)(=O)Cl.OCCBr>C(N(CC)CC)C>CCS(=O)(=O)OCCBr

To train/val/test CSV files with rxn_smiles column:
  rxn_smiles,rxn_class
  CCOCC.CCS(=O)(=O)Cl.OCCBr>>CCS(=O)(=O)OCCBr,0

Usage:
  python scripts/prepare_uspto_full.py \
      --input Data/USPTO/USPTO_FULL.csv \
      --output_dir Data/uspto_full \
      --train_ratio 0.8 --val_ratio 0.1
"""

import argparse
import csv
import os
import random
from collections import Counter

from rdkit import Chem


def parse_full_reaction(reaction_str: str) -> str | None:
    """
    Parse USPTO-FULL reaction format: reactants>reagent>products
    Returns rxn_smiles format: reactants>>products, or None if invalid.
    """
    parts = reaction_str.split('>')
    if len(parts) != 3:
        return None

    reactants_str = parts[0].strip()
    products_str = parts[2].strip()

    if not reactants_str or not products_str:
        return None

    # Validate and canonicalize reactants
    reactant_parts = [s.strip() for s in reactants_str.split('.') if s.strip()]
    if len(reactant_parts) < 2:
        return None  # Need at least 2 reactants for link prediction

    canon_reactants = []
    for r in reactant_parts:
        mol = Chem.MolFromSmiles(r)
        if mol is None:
            return None
        canon_reactants.append(Chem.MolToSmiles(mol))

    # Validate products
    product_parts = [s.strip() for s in products_str.split('.') if s.strip()]
    canon_products = []
    for p in product_parts:
        mol = Chem.MolFromSmiles(p)
        if mol is None:
            continue  # Skip invalid products but don't discard reaction
        canon_products.append(Chem.MolToSmiles(mol))

    if not canon_products:
        return None

    rxn_smiles = '.'.join(canon_reactants) + '>>' + '.'.join(canon_products)
    return rxn_smiles


def main():
    parser = argparse.ArgumentParser(description='Prepare USPTO-FULL for V3 training')
    parser.add_argument('--input', type=str, default='Data/USPTO/USPTO_FULL.csv')
    parser.add_argument('--output_dir', type=str, default='Data/uspto_full')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_reactions', type=int, default=0,
                        help='Max reactions to process (0 = all)')
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading {args.input}...")
    valid_reactions = []
    n_total = 0
    n_skipped_format = 0
    n_skipped_reactants = 0
    n_skipped_invalid = 0

    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if n_total % 200000 == 0:
                print(f"  Processed {n_total:,} rows, {len(valid_reactions):,} valid...")

            if args.max_reactions > 0 and len(valid_reactions) >= args.max_reactions:
                break

            reaction_str = row.get('reactions', '')
            if not reaction_str:
                n_skipped_format += 1
                continue

            rxn_smiles = parse_full_reaction(reaction_str)
            if rxn_smiles is None:
                parts = reaction_str.split('>')
                if len(parts) != 3:
                    n_skipped_format += 1
                else:
                    reactant_parts = [s.strip() for s in parts[0].split('.') if s.strip()]
                    if len(reactant_parts) < 2:
                        n_skipped_reactants += 1
                    else:
                        n_skipped_invalid += 1
                continue

            valid_reactions.append(rxn_smiles)

    print(f"\nTotal rows: {n_total:,}")
    print(f"Valid reactions (>=2 reactants): {len(valid_reactions):,}")
    print(f"Skipped (format): {n_skipped_format:,}")
    print(f"Skipped (<2 reactants): {n_skipped_reactants:,}")
    print(f"Skipped (invalid SMILES): {n_skipped_invalid:,}")

    # Deduplicate
    unique_reactions = list(set(valid_reactions))
    print(f"Unique reactions: {len(unique_reactions):,} (dedup removed {len(valid_reactions) - len(unique_reactions):,})")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(unique_reactions)

    n = len(unique_reactions)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_rxns = unique_reactions[:n_train]
    val_rxns = unique_reactions[n_train:n_train + n_val]
    test_rxns = unique_reactions[n_train + n_val:]

    print(f"\nSplit: train={len(train_rxns):,}, val={len(val_rxns):,}, test={len(test_rxns):,}")

    # Write CSV files
    for split_name, rxns in [('train', train_rxns), ('val', val_rxns), ('test', test_rxns)]:
        out_path = os.path.join(args.output_dir, f'{split_name}.csv')
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rxn_smiles', 'rxn_class'])
            for rxn in rxns:
                writer.writerow([rxn, 0])
        print(f"  Wrote {out_path}: {len(rxns):,} reactions")

    # Statistics
    all_reactants = set()
    for rxn in unique_reactions:
        parts = rxn.split('>>')
        for r in parts[0].split('.'):
            all_reactants.add(r)
    print(f"\nUnique reactant molecules: {len(all_reactants):,}")

    # Count reactants per reaction
    n_reactants = [len(rxn.split('>>')[0].split('.')) for rxn in unique_reactions]
    cnt = Counter(n_reactants)
    print("Reactants per reaction distribution:")
    for k in sorted(cnt.keys()):
        print(f"  {k} reactants: {cnt[k]:,} ({100*cnt[k]/len(unique_reactions):.1f}%)")

    print("\nDone!")


if __name__ == '__main__':
    main()
