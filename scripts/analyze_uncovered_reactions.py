#!/usr/bin/env python
"""Analyze uncovered reactions in USPTO 50K.

Identifies what types of reactions the 71 SMARTS templates fail to cover,
classifies them by Schneider reaction class, functional group changes,
and proposes priority template expansion candidates.

Usage:
    python scripts/analyze_uncovered_reactions.py

Output:
    docs/uncovered_reactions_analysis.md
"""

import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import Draw
from rdkit.DataStructs import TanimotoSimilarity

# Suppress RDKit warnings for cleaner output
RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "uspto"
TEMPLATE_PATH = PROJECT_ROOT / "template" / "data" / "templates.txt"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "uncovered_reactions_analysis.md"

# Schneider 10-class names
SCHNEIDER_CLASSES = {
    1: "Heteroatom alkylation & arylation",
    2: "Acylation & related",
    3: "C-C bond formation",
    4: "Heterocycle formation",
    5: "Protections",
    6: "Deprotections",
    7: "Reductions",
    8: "Oxidations",
    9: "Functional group interconversion (FGI)",
    10: "Functional group addition (FGA)",
}

# Subclass definitions for heuristic classification of uncovered reactions
# These are based on common reaction patterns in medicinal chemistry
FUNCTIONAL_GROUP_SMARTS = {
    # Common functional groups to track appearance/disappearance
    "primary_amine": "[NH2;!$(NC=O)]",
    "secondary_amine": "[NH1;!$(NC=O);!$(Nc)]",
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "alcohol": "[OX2H][CX4]",
    "phenol": "[OX2H][c]",
    "aldehyde": "[CX3H1](=O)",
    "ketone": "[#6][CX3](=O)[#6]",
    "nitro": "[N+](=O)[O-]",
    "nitrile": "[CX2]#[NX1]",
    "halide_F": "[CX4,c][F]",
    "halide_Cl": "[CX4,c][Cl]",
    "halide_Br": "[CX4,c][Br]",
    "halide_I": "[CX4,c][I]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])([NX3])",
    "sulfonate": "[SX4](=[OX1])(=[OX1])([OX2])",
    "Boc": "C(=O)OC(C)(C)C",
    "Cbz": "C(=O)OCc1ccccc1",
    "Fmoc": "C(=O)OCC1c2ccccc2-c2ccccc21",
    "benzyl": "[CH2]c1ccccc1",
    "TBS": "[Si](C)(C)C(C)(C)C",
    "boronic_acid": "[BX3](O)(O)",
    "boronate_ester": "[BX3]([OX2])([OX2])",
    "vinyl": "[CX3]=[CX3]",
    "aryl_halide": "[c][F,Cl,Br,I]",
    "epoxide": "C1OC1",
    "azide": "[N-]=[N+]=N",
    "thiol": "[SH]",
    "phosphonate": "[PX4](=O)([OX2])([OX2])",
}

# Known coupling reaction patterns (product-side SMARTS heuristics)
COUPLING_PATTERNS = {
    "Suzuki": ("[c]-[c]", "[c][BX3]", "[c][Cl,Br,I]"),  # biaryl from boronic acid + aryl halide
    "Heck": ("[CX3]=[CX3][c]", None, "[c][Cl,Br,I]"),    # vinyl-aryl
    "Sonogashira": ("[CX2]#[CX2][c]", "[C]#[CH]", "[c][Cl,Br,I]"),  # alkynyl-aryl
    "Buchwald-Hartwig": ("[c][NX3]", "[NX3;H]", "[c][Cl,Br,I]"),  # C-N cross-coupling
    "Negishi": ("[c]-[c,C]", "[Zn]", "[c][Cl,Br,I]"),
    "Stille": ("[c]-[c,C]", "[Sn]", "[c][Cl,Br,I]"),
    "Wittig/HWE": ("[CX3]=[CX3]", "[PX4]", "[CX3]=[O]"),
}


def load_all_reactions() -> pd.DataFrame:
    """Load train + val + test splits into one DataFrame."""
    dfs = []
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.csv"
        df = pd.read_csv(path)
        df["split"] = split
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} reactions ({', '.join(f'{s}={len(d)}' for s, d in zip(['train','val','test'], dfs))})")
    return combined


def load_templates() -> list[str]:
    """Load SMARTS templates from file."""
    templates = []
    with open(TEMPLATE_PATH) as f:
        for line in f:
            t = line.strip()
            if t:
                templates.append(t)
    print(f"Loaded {len(templates)} templates")
    return templates


def parse_rxn_smiles(rxn_smiles: str):
    """Parse reaction SMILES into (reactants_list, products_list).

    Returns (list[str], list[str]) of individual SMILES.
    """
    parts = rxn_smiles.split(">>")
    if len(parts) != 2:
        return None, None
    reactants = parts[0].split(".")
    products = parts[1].split(".")
    return reactants, products


def apply_template_to_reaction(template_smarts: str, reactant_smiles_list: list[str],
                                product_smiles: str) -> tuple[bool, float]:
    """Check if a template produces the target product from reactants.

    Returns (exact_match, best_tanimoto).
    """
    try:
        rxn = AllChem.ReactionFromSmarts(template_smarts)
        AllChem.ChemicalReaction.Initialize(rxn)
    except Exception:
        return False, 0.0

    n_reactants = rxn.GetNumReactantTemplates()

    # Parse reactant molecules
    reactant_mols = []
    for smi in reactant_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False, 0.0
        reactant_mols.append(mol)

    # Parse product
    prod_mol = Chem.MolFromSmiles(product_smiles)
    if prod_mol is None:
        return False, 0.0
    prod_fp = AllChem.GetMorganFingerprintAsBitVect(prod_mol, 2, nBits=2048)
    prod_canon = Chem.MolToSmiles(prod_mol)

    best_tanimoto = 0.0

    if n_reactants == 1:
        # Try each reactant
        for mol in reactant_mols:
            if not rxn.IsMoleculeReactant(mol):
                continue
            try:
                products_sets = rxn.RunReactants((mol,), 5)
            except Exception:
                continue
            for ps in products_sets:
                for p in ps:
                    try:
                        p = Chem.RemoveHs(p, updateExplicitCount=True)
                        p_smi = Chem.MolToSmiles(p)
                        p_smi = p_smi.replace("[C]", "C").replace("[N]", "N").replace("[CH]", "C")
                        p2 = Chem.MolFromSmiles(p_smi)
                        if p2 is None:
                            continue
                        p_canon = Chem.MolToSmiles(p2)
                        if p_canon == prod_canon:
                            return True, 1.0
                        p_fp = AllChem.GetMorganFingerprintAsBitVect(p2, 2, nBits=2048)
                        sim = TanimotoSimilarity(prod_fp, p_fp)
                        best_tanimoto = max(best_tanimoto, sim)
                    except Exception:
                        continue

    elif n_reactants == 2:
        # Try all pairs
        from itertools import permutations
        for i, j in permutations(range(len(reactant_mols)), 2):
            if not rxn.IsMoleculeReactant(reactant_mols[i]):
                continue
            try:
                products_sets = rxn.RunReactants((reactant_mols[i], reactant_mols[j]), 5)
            except Exception:
                continue
            for ps in products_sets:
                for p in ps:
                    try:
                        p = Chem.RemoveHs(p, updateExplicitCount=True)
                        p_smi = Chem.MolToSmiles(p)
                        p_smi = p_smi.replace("[C]", "C").replace("[N]", "N").replace("[CH]", "C")
                        p2 = Chem.MolFromSmiles(p_smi)
                        if p2 is None:
                            continue
                        p_canon = Chem.MolToSmiles(p2)
                        if p_canon == prod_canon:
                            return True, 1.0
                        p_fp = AllChem.GetMorganFingerprintAsBitVect(p2, 2, nBits=2048)
                        sim = TanimotoSimilarity(prod_fp, p_fp)
                        best_tanimoto = max(best_tanimoto, sim)
                    except Exception:
                        continue

    return False, best_tanimoto


def evaluate_coverage_batch(reactions_df: pd.DataFrame, templates: list[str]) -> pd.DataFrame:
    """Evaluate template coverage for all reactions. Returns df with coverage columns."""
    n = len(reactions_df)
    covered = np.zeros(n, dtype=bool)
    best_sim = np.zeros(n, dtype=np.float32)
    best_template = np.full(n, -1, dtype=np.int32)

    t0 = time.perf_counter()

    for idx, row in reactions_df.iterrows():
        rxn = row["rxn_smiles"]
        reactants_smi, products_smi = parse_rxn_smiles(rxn)
        if reactants_smi is None or products_smi is None:
            continue

        # Take the main product (largest fragment if multiple)
        main_product = max(products_smi, key=len)

        row_best_sim = 0.0
        row_best_tmpl = -1
        row_covered = False

        for t_idx, tmpl in enumerate(templates):
            exact, sim = apply_template_to_reaction(tmpl, reactants_smi, main_product)
            if exact:
                row_covered = True
                row_best_sim = 1.0
                row_best_tmpl = t_idx
                break
            if sim > row_best_sim:
                row_best_sim = sim
                row_best_tmpl = t_idx

        covered[idx] = row_covered or (row_best_sim >= 0.8)
        best_sim[idx] = row_best_sim
        best_template[idx] = row_best_tmpl

        if (idx + 1) % 2000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / elapsed
            eta = (n - idx - 1) / rate
            print(f"  [{idx+1}/{n}] {elapsed:.0f}s elapsed, {rate:.0f} rxn/s, ETA {eta:.0f}s "
                  f"| covered so far: {covered[:idx+1].sum()}/{idx+1} ({covered[:idx+1].mean()*100:.1f}%)")

    elapsed = time.perf_counter() - t0
    print(f"Coverage evaluation: {elapsed:.1f}s for {n} reactions")

    reactions_df = reactions_df.copy()
    reactions_df["covered"] = covered
    reactions_df["best_sim"] = best_sim
    reactions_df["best_template"] = best_template
    return reactions_df


def _eval_single_reaction(args):
    """Worker function for parallel template evaluation."""
    row_idx, rxn_smiles, templates = args
    reactants_smi, products_smi = parse_rxn_smiles(rxn_smiles)
    if reactants_smi is None or products_smi is None:
        return row_idx, False, 0.0, -1

    main_product = max(products_smi, key=len)

    best_sim = 0.0
    best_tmpl = -1
    is_covered = False

    for t_idx, tmpl in enumerate(templates):
        exact, sim = apply_template_to_reaction(tmpl, reactants_smi, main_product)
        if exact:
            is_covered = True
            best_sim = 1.0
            best_tmpl = t_idx
            break
        if sim > best_sim:
            best_sim = sim
            best_tmpl = t_idx

    is_covered = is_covered or (best_sim >= 0.8)
    return row_idx, is_covered, best_sim, best_tmpl


def evaluate_coverage_parallel(reactions_df: pd.DataFrame, templates: list[str],
                                num_workers: int = 16) -> pd.DataFrame:
    """Parallel version of coverage evaluation using ProcessPoolExecutor."""
    n = len(reactions_df)
    covered = np.zeros(n, dtype=bool)
    best_sim = np.zeros(n, dtype=np.float32)
    best_template = np.full(n, -1, dtype=np.int32)

    t0 = time.perf_counter()

    # Prepare args: pass template strings (picklable), not RDKit objects
    args_list = [(idx, row["rxn_smiles"], templates)
                 for idx, row in reactions_df.iterrows()]

    done = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit in chunks for better progress reporting
        futures = {executor.submit(_eval_single_reaction, arg): arg[0] for arg in args_list}

        for future in as_completed(futures):
            row_idx, is_cov, sim, tmpl = future.result()
            covered[row_idx] = is_cov
            best_sim[row_idx] = sim
            best_template[row_idx] = tmpl
            done += 1
            if done % 5000 == 0:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed
                eta = (n - done) / rate if rate > 0 else 0
                print(f"  [{done}/{n}] {elapsed:.0f}s, {rate:.0f} rxn/s, ETA {eta:.0f}s "
                      f"| covered: {covered.sum()}")

    elapsed = time.perf_counter() - t0
    print(f"Parallel coverage evaluation: {elapsed:.1f}s for {n} reactions ({num_workers} workers)")

    reactions_df = reactions_df.copy()
    reactions_df["covered"] = covered
    reactions_df["best_sim"] = best_sim
    reactions_df["best_template"] = best_template
    return reactions_df


def analyze_functional_groups(rxn_smiles: str) -> dict:
    """Analyze functional group changes in a reaction.

    Returns dict with:
        - fg_gained: list of FG names gained in product
        - fg_lost: list of FG names lost from reactants
        - mw_change: MW of product - sum of reactant MWs
        - n_reactants: number of reactants
        - reactant_mw: total reactant MW
        - product_mw: product MW
    """
    reactants_smi, products_smi = parse_rxn_smiles(rxn_smiles)
    if reactants_smi is None or products_smi is None:
        return {}

    main_product = max(products_smi, key=len)

    # Parse molecules
    reactant_mols = [Chem.MolFromSmiles(s) for s in reactants_smi]
    reactant_mols = [m for m in reactant_mols if m is not None]
    prod_mol = Chem.MolFromSmiles(main_product)

    if not reactant_mols or prod_mol is None:
        return {}

    # MW
    reactant_mw = sum(Descriptors.MolWt(m) for m in reactant_mols)
    product_mw = Descriptors.MolWt(prod_mol)

    # Functional group analysis
    fg_in_reactants = set()
    fg_in_product = set()

    for fg_name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue

        # Check reactants
        for mol in reactant_mols:
            if mol.HasSubstructMatch(pattern):
                fg_in_reactants.add(fg_name)
                break

        # Check product
        if prod_mol.HasSubstructMatch(pattern):
            fg_in_product.add(fg_name)

    fg_gained = fg_in_product - fg_in_reactants
    fg_lost = fg_in_reactants - fg_in_product

    return {
        "n_reactants": len(reactants_smi),
        "reactant_mw": reactant_mw,
        "product_mw": product_mw,
        "mw_change": product_mw - reactant_mw,
        "mw_ratio": product_mw / max(reactant_mw, 0.1),
        "fg_gained": list(fg_gained),
        "fg_lost": list(fg_lost),
        "fg_in_reactants": list(fg_in_reactants),
        "fg_in_product": list(fg_in_product),
    }


def classify_reaction_heuristic(rxn_smiles: str, rxn_class: int, fg_info: dict) -> str:
    """Heuristic subclass of a reaction based on FG changes and class.

    Returns a descriptive string like "N-alkylation", "Boc_deprotection", etc.
    """
    if not fg_info:
        return "unknown"

    fg_lost = set(fg_info.get("fg_lost", []))
    fg_gained = set(fg_info.get("fg_gained", []))
    fg_reactants = set(fg_info.get("fg_in_reactants", []))
    fg_products = set(fg_info.get("fg_in_product", []))
    mw_change = fg_info.get("mw_change", 0)
    n_reactants = fg_info.get("n_reactants", 1)

    # --- Class 1: Heteroatom alkylation & arylation ---
    if rxn_class == 1:
        if "aryl_halide" in fg_reactants and ("secondary_amine" in fg_gained or "primary_amine" in fg_lost):
            return "N-arylation (Buchwald-Hartwig type)"
        if "aryl_halide" in fg_reactants and "phenol" in fg_reactants:
            return "O-arylation"
        if any(h in fg_lost for h in ["halide_Cl", "halide_Br", "halide_I"]):
            if "primary_amine" in fg_reactants or "secondary_amine" in fg_reactants:
                return "N-alkylation"
            if "alcohol" in fg_reactants or "phenol" in fg_reactants:
                return "O-alkylation"
            if "thiol" in fg_reactants:
                return "S-alkylation"
            return "Alkylation (unspec.)"
        if "epoxide" in fg_lost:
            return "Epoxide opening"
        return "Heteroatom alkylation/arylation (unspec.)"

    # --- Class 2: Acylation ---
    if rxn_class == 2:
        if "amide" in fg_gained and ("carboxylic_acid" in fg_lost or "ester" in fg_lost):
            return "Amide coupling"
        if "amide" in fg_gained:
            return "Amidation"
        if "ester" in fg_gained:
            return "Esterification"
        if "sulfonamide" in fg_gained:
            return "Sulfonamide formation"
        if "carboxylic_acid" in fg_lost:
            return "Acylation (from acid)"
        return "Acylation (unspec.)"

    # --- Class 3: C-C bond formation ---
    if rxn_class == 3:
        if "boronic_acid" in fg_lost or "boronate_ester" in fg_lost:
            return "Suzuki coupling"
        if "aryl_halide" in fg_reactants:
            if "vinyl" in fg_reactants:
                return "Heck coupling"
            if "nitrile" in fg_reactants:
                return "Cyanation"
            if n_reactants >= 2:
                return "C-C cross-coupling (aryl halide)"
        if any(h in fg_reactants for h in ["halide_Cl", "halide_Br", "halide_I"]):
            return "C-C coupling (alkyl halide)"
        if "phosphonate" in fg_reactants:
            return "Wittig/HWE olefination"
        if "aldehyde" in fg_reactants or "ketone" in fg_reactants:
            return "Aldol/condensation"
        return "C-C bond formation (unspec.)"

    # --- Class 4: Heterocycle formation ---
    if rxn_class == 4:
        return "Heterocycle formation"

    # --- Class 5: Protections ---
    if rxn_class == 5:
        if "Boc" in fg_gained:
            return "Boc protection"
        if "Cbz" in fg_gained:
            return "Cbz protection"
        if "TBS" in fg_gained:
            return "TBS protection"
        if "Fmoc" in fg_gained:
            return "Fmoc protection"
        if "benzyl" in fg_gained and ("alcohol" in fg_lost or "phenol" in fg_lost):
            return "Benzyl protection (O)"
        if "ester" in fg_gained and "carboxylic_acid" in fg_lost:
            return "Acid protection (ester)"
        if "amide" in fg_gained and "primary_amine" in fg_lost:
            return "Amine protection (amide)"
        return "Protection (unspec.)"

    # --- Class 6: Deprotections ---
    if rxn_class == 6:
        if "Boc" in fg_lost:
            return "Boc deprotection"
        if "Cbz" in fg_lost:
            return "Cbz deprotection"
        if "TBS" in fg_lost:
            return "TBS/silyl deprotection"
        if "Fmoc" in fg_lost:
            return "Fmoc deprotection"
        if "benzyl" in fg_lost:
            return "Benzyl deprotection"
        if "ester" in fg_lost and "carboxylic_acid" in fg_gained:
            return "Ester hydrolysis (deprotection)"
        if mw_change < -50:
            return "Deprotection (large group removal)"
        return "Deprotection (unspec.)"

    # --- Class 7: Reductions ---
    if rxn_class == 7:
        if "nitro" in fg_lost and "primary_amine" in fg_gained:
            return "Nitro reduction (-> amine)"
        if "nitrile" in fg_lost and "primary_amine" in fg_gained:
            return "Nitrile reduction (-> amine)"
        if "aldehyde" in fg_lost and "alcohol" in fg_gained:
            return "Aldehyde reduction (-> alcohol)"
        if "ketone" in fg_lost and "alcohol" in fg_gained:
            return "Ketone reduction (-> alcohol)"
        if "amide" in fg_lost and "secondary_amine" in fg_gained:
            return "Amide reduction (-> amine)"
        if "ester" in fg_lost and "alcohol" in fg_gained:
            return "Ester reduction (-> alcohol)"
        if "vinyl" in fg_lost:
            return "Olefin reduction (hydrogenation)"
        if "azide" in fg_lost and "primary_amine" in fg_gained:
            return "Azide reduction (-> amine)"
        if "primary_amine" in fg_gained:
            return "Reduction (-> amine)"
        if "alcohol" in fg_gained:
            return "Reduction (-> alcohol)"
        return "Reduction (unspec.)"

    # --- Class 8: Oxidations ---
    if rxn_class == 8:
        if "alcohol" in fg_lost and ("aldehyde" in fg_gained or "ketone" in fg_gained):
            return "Alcohol oxidation (-> carbonyl)"
        if "alcohol" in fg_lost and "carboxylic_acid" in fg_gained:
            return "Alcohol oxidation (-> acid)"
        if "thiol" in fg_lost:
            return "Thiol oxidation"
        if "sulfonate" in fg_gained:
            return "Sulfur oxidation"
        return "Oxidation (unspec.)"

    # --- Class 9: FGI ---
    if rxn_class == 9:
        if "aryl_halide" in fg_reactants and "nitrile" in fg_gained:
            return "FGI: Halide -> Nitrile"
        if "halide_F" in fg_gained and ("halide_Cl" in fg_lost or "halide_Br" in fg_lost):
            return "FGI: Halide exchange (-> F)"
        if "halide_Cl" in fg_gained and ("halide_Br" in fg_lost or "halide_I" in fg_lost):
            return "FGI: Halide exchange (-> Cl)"
        if "primary_amine" in fg_lost and "azide" in fg_gained:
            return "FGI: Amine -> Azide"
        if "carboxylic_acid" in fg_reactants and "amide" in fg_gained:
            return "FGI: Acid -> Amide"
        if "alcohol" in fg_lost and any(h in fg_gained for h in ["halide_Cl", "halide_Br"]):
            return "FGI: Alcohol -> Halide"
        if any(h in fg_gained for h in ["halide_Cl", "halide_Br", "halide_I"]):
            return "FGI: Halogenation"
        return "FGI (unspec.)"

    # --- Class 10: FGA ---
    if rxn_class == 10:
        return "Functional group addition"

    return "Unclassified"


def analyze_atom_changes(rxn_smiles: str) -> dict:
    """Analyze atom-level changes between reactants and product."""
    reactants_smi, products_smi = parse_rxn_smiles(rxn_smiles)
    if reactants_smi is None or products_smi is None:
        return {}

    main_product = max(products_smi, key=len)

    # Count atoms in reactants vs product
    reactant_atoms = Counter()
    for smi in reactants_smi:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            reactant_atoms[atom.GetSymbol()] += 1

    prod_mol = Chem.MolFromSmiles(main_product)
    if prod_mol is None:
        return {}
    product_atoms = Counter()
    for atom in prod_mol.GetAtoms():
        product_atoms[atom.GetSymbol()] += 1

    # Compute changes
    all_symbols = set(reactant_atoms.keys()) | set(product_atoms.keys())
    atom_changes = {}
    for sym in all_symbols:
        delta = product_atoms.get(sym, 0) - reactant_atoms.get(sym, 0)
        if delta != 0:
            atom_changes[sym] = delta

    return atom_changes


def compute_reaction_fingerprint(rxn_smiles: str) -> str:
    """Compute a coarse 'fingerprint' of what the reaction does.

    Returns a string like "lost:Boc,halide_Br|gained:primary_amine"
    """
    fg_info = analyze_functional_groups(rxn_smiles)
    if not fg_info:
        return "unknown"

    lost = sorted(fg_info.get("fg_lost", []))
    gained = sorted(fg_info.get("fg_gained", []))

    parts = []
    if lost:
        parts.append("lost:" + ",".join(lost))
    if gained:
        parts.append("gained:" + ",".join(gained))

    if not parts:
        # Use MW change as fallback
        mw_change = fg_info.get("mw_change", 0)
        if mw_change > 20:
            parts.append("MW+")
        elif mw_change < -20:
            parts.append("MW-")
        else:
            parts.append("MW~0")

    return "|".join(parts)


def generate_report(df: pd.DataFrame, templates: list[str]) -> str:
    """Generate the markdown analysis report."""
    lines = []

    uncov = df[~df["covered"]].copy()
    cov = df[df["covered"]].copy()
    total = len(df)
    n_uncov = len(uncov)
    n_cov = len(cov)

    lines.append("# Uncovered Reactions Analysis: USPTO 50K")
    lines.append("")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Total reactions**: {total}")
    lines.append(f"**Covered (exact/near >= 0.8)**: {n_cov} ({n_cov/total*100:.1f}%)")
    lines.append(f"**Uncovered**: {n_uncov} ({n_uncov/total*100:.1f}%)")
    lines.append(f"**Templates**: {len(templates)}")
    lines.append("")

    # ================================================================
    # Section 1: Coverage by reaction class
    # ================================================================
    lines.append("## 1. Coverage by Reaction Class")
    lines.append("")
    lines.append("| Class | Name | Total | Covered | Uncovered | Coverage % |")
    lines.append("|------:|------|------:|--------:|----------:|-----------:|")

    class_stats = []
    for cls in sorted(df["class"].unique()):
        cls_df = df[df["class"] == cls]
        cls_cov = cls_df["covered"].sum()
        cls_total = len(cls_df)
        cls_uncov = cls_total - cls_cov
        name = SCHNEIDER_CLASSES.get(cls, f"Unknown ({cls})")
        pct = cls_cov / cls_total * 100 if cls_total > 0 else 0
        lines.append(f"| {cls} | {name} | {cls_total} | {cls_cov} | {cls_uncov} | {pct:.1f}% |")
        class_stats.append((cls, name, cls_total, cls_cov, cls_uncov, pct))

    lines.append("")

    # Highlight completely/nearly uncovered classes
    lines.append("### Key Observations")
    lines.append("")
    for cls, name, tot, cov_n, uncov_n, pct in class_stats:
        if pct < 1.0:
            lines.append(f"- **Class {cls} ({name})**: Essentially zero coverage ({pct:.1f}%). "
                        f"All {uncov_n} reactions are uncovered.")
        elif pct < 15.0:
            lines.append(f"- **Class {cls} ({name})**: Very low coverage ({pct:.1f}%). "
                        f"{uncov_n} of {tot} reactions uncovered.")
    lines.append("")

    # ================================================================
    # Section 2: Uncovered reactions by number of reactants
    # ================================================================
    lines.append("## 2. Uncovered Reactions by Number of Reactants")
    lines.append("")

    # Count reactants
    def count_reactants(rxn):
        parts = rxn.split(">>")
        if len(parts) != 2:
            return 0
        return len(parts[0].split("."))

    uncov["n_reactants"] = uncov["rxn_smiles"].apply(count_reactants)

    lines.append("| # Reactants | Uncovered Count | % of All Uncovered |")
    lines.append("|------------:|----------------:|-------------------:|")
    for nr in sorted(uncov["n_reactants"].unique()):
        cnt = (uncov["n_reactants"] == nr).sum()
        lines.append(f"| {nr} | {cnt} | {cnt/n_uncov*100:.1f}% |")
    lines.append("")

    # ================================================================
    # Section 3: Detailed FG analysis of uncovered reactions
    # ================================================================
    lines.append("## 3. Functional Group Change Analysis (Uncovered Reactions)")
    lines.append("")

    print("Analyzing functional groups for uncovered reactions...")
    t0 = time.perf_counter()

    fg_results = []
    subclass_labels = []
    reaction_fps = []

    for idx, row in uncov.iterrows():
        fg = analyze_functional_groups(row["rxn_smiles"])
        fg_results.append(fg)

        subcls = classify_reaction_heuristic(row["rxn_smiles"], row["class"], fg)
        subclass_labels.append(subcls)

        rfp = compute_reaction_fingerprint(row["rxn_smiles"])
        reaction_fps.append(rfp)

    uncov = uncov.copy()
    uncov["subclass"] = subclass_labels
    uncov["rxn_fingerprint"] = reaction_fps

    elapsed = time.perf_counter() - t0
    print(f"  FG analysis: {elapsed:.1f}s for {n_uncov} reactions")

    # MW change distribution
    mw_changes = [fg.get("mw_change", 0) for fg in fg_results if fg]
    if mw_changes:
        lines.append("### 3a. Molecular Weight Change Distribution")
        lines.append("")
        lines.append(f"- Mean MW change: {np.mean(mw_changes):.1f} Da")
        lines.append(f"- Median MW change: {np.median(mw_changes):.1f} Da")
        lines.append(f"- Std: {np.std(mw_changes):.1f} Da")
        lines.append(f"- Range: [{np.min(mw_changes):.0f}, {np.max(mw_changes):.0f}] Da")
        lines.append("")

        # Buckets
        lines.append("| MW Change Range | Count | % |")
        lines.append("|-----------------|------:|--:|")
        mw_arr = np.array(mw_changes)
        buckets = [
            ("< -200 Da (large loss)", mw_arr < -200),
            ("-200 to -100 Da", (mw_arr >= -200) & (mw_arr < -100)),
            ("-100 to -50 Da", (mw_arr >= -100) & (mw_arr < -50)),
            ("-50 to -10 Da", (mw_arr >= -50) & (mw_arr < -10)),
            ("-10 to +10 Da (isomerization)", (mw_arr >= -10) & (mw_arr <= 10)),
            ("+10 to +50 Da", (mw_arr > 10) & (mw_arr <= 50)),
            ("+50 to +100 Da", (mw_arr > 50) & (mw_arr <= 100)),
            ("+100 to +200 Da", (mw_arr > 100) & (mw_arr <= 200)),
            ("> +200 Da (large addition)", mw_arr > 200),
        ]
        for label, mask in buckets:
            cnt = mask.sum()
            lines.append(f"| {label} | {cnt} | {cnt/len(mw_arr)*100:.1f}% |")
        lines.append("")

    # FG lost/gained frequency
    fg_lost_counter = Counter()
    fg_gained_counter = Counter()
    for fg in fg_results:
        if fg:
            for g in fg.get("fg_lost", []):
                fg_lost_counter[g] += 1
            for g in fg.get("fg_gained", []):
                fg_gained_counter[g] += 1

    lines.append("### 3b. Most Commonly Lost Functional Groups (Top 20)")
    lines.append("")
    lines.append("| Functional Group | Count | % of Uncovered |")
    lines.append("|------------------|------:|---------------:|")
    for fg_name, cnt in fg_lost_counter.most_common(20):
        lines.append(f"| {fg_name} | {cnt} | {cnt/n_uncov*100:.1f}% |")
    lines.append("")

    lines.append("### 3c. Most Commonly Gained Functional Groups (Top 20)")
    lines.append("")
    lines.append("| Functional Group | Count | % of Uncovered |")
    lines.append("|------------------|------:|---------------:|")
    for fg_name, cnt in fg_gained_counter.most_common(20):
        lines.append(f"| {fg_name} | {cnt} | {cnt/n_uncov*100:.1f}% |")
    lines.append("")

    # ================================================================
    # Section 4: Subclass distribution of uncovered reactions
    # ================================================================
    lines.append("## 4. Reaction Subclass Distribution (Uncovered)")
    lines.append("")
    lines.append("Heuristic classification of uncovered reactions by functional group changes:")
    lines.append("")

    subclass_counter = Counter(subclass_labels)

    lines.append("| # | Reaction Subclass | Count | % of Uncovered | Schneider Class |")
    lines.append("|--:|-------------------|------:|---------------:|----------------:|")

    # Get class for each subclass
    subclass_to_classes = defaultdict(Counter)
    for subcls, cls in zip(subclass_labels, uncov["class"]):
        subclass_to_classes[subcls][cls] += 1

    for rank, (subcls, cnt) in enumerate(subclass_counter.most_common(40), 1):
        top_class = subclass_to_classes[subcls].most_common(1)[0][0]
        lines.append(f"| {rank} | {subcls} | {cnt} | {cnt/n_uncov*100:.1f}% | {top_class} |")
    lines.append("")

    # ================================================================
    # Section 5: Per-class detailed breakdown
    # ================================================================
    lines.append("## 5. Per-Class Detailed Breakdown of Uncovered Reactions")
    lines.append("")

    for cls in sorted(uncov["class"].unique()):
        cls_uncov = uncov[uncov["class"] == cls]
        cls_name = SCHNEIDER_CLASSES.get(cls, f"Unknown ({cls})")
        lines.append(f"### Class {cls}: {cls_name} ({len(cls_uncov)} uncovered)")
        lines.append("")

        # Subclass distribution within this class
        cls_subclass_counter = Counter(cls_uncov["subclass"])
        lines.append("| Subclass | Count | % of Class |")
        lines.append("|----------|------:|-----------:|")
        for subcls, cnt in cls_subclass_counter.most_common(15):
            lines.append(f"| {subcls} | {cnt} | {cnt/len(cls_uncov)*100:.1f}% |")
        lines.append("")

        # Example reactions (up to 5)
        lines.append("**Example uncovered reactions:**")
        lines.append("```")
        for i, (_, row) in enumerate(cls_uncov.head(5).iterrows()):
            lines.append(f"  {row['rxn_smiles']}")
        lines.append("```")
        lines.append("")

    # ================================================================
    # Section 6: High-frequency uncovered reaction patterns (template candidates)
    # ================================================================
    lines.append("## 6. Template Expansion Candidates")
    lines.append("")
    lines.append("High-frequency uncovered reaction patterns that should be prioritized "
                "for template expansion:")
    lines.append("")

    # Group by (class, subclass) and count
    expansion_candidates = []
    for (cls, subcls), group in uncov.groupby(["class", "subclass"]):
        expansion_candidates.append({
            "class": cls,
            "class_name": SCHNEIDER_CLASSES.get(cls, "?"),
            "subclass": subcls,
            "count": len(group),
            "pct": len(group) / total * 100,
        })

    expansion_candidates.sort(key=lambda x: -x["count"])

    lines.append("### Priority 1: Highest Volume Missing Reaction Types")
    lines.append("")
    lines.append("| # | Class | Subclass | Count | % of All | Cumulative % |")
    lines.append("|--:|------:|----------|------:|---------:|-------------:|")

    cum_pct = 0
    for rank, cand in enumerate(expansion_candidates[:30], 1):
        cum_pct += cand["pct"]
        lines.append(f"| {rank} | {cand['class']} ({cand['class_name'][:20]}) | "
                    f"{cand['subclass']} | {cand['count']} | {cand['pct']:.1f}% | {cum_pct:.1f}% |")
    lines.append("")

    # Estimate coverage improvement
    lines.append("### Estimated Coverage Improvement from Template Expansion")
    lines.append("")

    current_coverage = n_cov / total * 100

    for top_n in [5, 10, 15, 20]:
        additional = sum(c["count"] for c in expansion_candidates[:top_n])
        new_coverage = (n_cov + additional) / total * 100
        lines.append(f"- **Adding top-{top_n} uncovered types** ({additional} reactions): "
                    f"coverage {current_coverage:.1f}% -> **{new_coverage:.1f}%** "
                    f"(+{new_coverage - current_coverage:.1f}pp)")

    lines.append("")
    lines.append(f"- Adding ALL uncovered types: coverage -> "
                f"{(n_cov + n_uncov) / total * 100:.1f}% (by definition)")
    lines.append("")

    # ================================================================
    # Section 7: Similarity distribution for uncovered reactions
    # ================================================================
    lines.append("## 7. Best Template Similarity Distribution (Uncovered Only)")
    lines.append("")
    lines.append("How close do uncovered reactions come to existing templates?")
    lines.append("")

    sims = uncov["best_sim"].values
    lines.append(f"- Mean best similarity: {np.mean(sims):.3f}")
    lines.append(f"- Median: {np.median(sims):.3f}")
    lines.append(f"- Std: {np.std(sims):.3f}")
    lines.append("")

    lines.append("| Similarity Range | Count | % | Interpretation |")
    lines.append("|------------------|------:|--:|----------------|")
    ranges = [
        (0.0, 0.2, "Completely different chemistry"),
        (0.2, 0.4, "Different reaction type"),
        (0.4, 0.6, "Somewhat related"),
        (0.6, 0.8, "Close but different product"),
    ]
    for lo, hi, interp in ranges:
        mask = (sims >= lo) & (sims < hi)
        cnt = mask.sum()
        lines.append(f"| [{lo:.1f}, {hi:.1f}) | {cnt} | {cnt/n_uncov*100:.1f}% | {interp} |")
    lines.append("")

    lines.append("Reactions with similarity 0.6-0.8 may be coverable by relaxing existing templates. "
                "Reactions below 0.4 require entirely new templates.")
    lines.append("")

    # ================================================================
    # Section 8: Recommendations
    # ================================================================
    lines.append("## 8. Recommendations for Template Expansion")
    lines.append("")

    # Top 5 recommendations
    top5 = expansion_candidates[:5]
    lines.append("### Immediate Priorities (Top 5)")
    lines.append("")
    for i, cand in enumerate(top5, 1):
        lines.append(f"{i}. **{cand['subclass']}** (Class {cand['class']}: {cand['class_name']})")
        lines.append(f"   - {cand['count']} reactions ({cand['pct']:.1f}% of dataset)")

        # Specific template suggestion
        if "deprotection" in cand['subclass'].lower() or "Deprotection" in cand['subclass']:
            lines.append(f"   - Suggestion: Add protecting group removal SMARTS "
                        f"(Boc/Cbz/benzyl/silyl/Fmoc cleavage patterns)")
        elif "reduction" in cand['subclass'].lower() or "Reduction" in cand['subclass']:
            lines.append(f"   - Suggestion: Add reduction SMARTS "
                        f"(nitro->amine, carbonyl->alcohol, catalytic hydrogenation)")
        elif "suzuki" in cand['subclass'].lower():
            lines.append(f"   - Suggestion: Add Suzuki coupling SMARTS "
                        f"([c:1][B](O)(O).[c:2][Cl,Br,I]>>[c:1][c:2])")
        elif "alkylation" in cand['subclass'].lower():
            lines.append(f"   - Suggestion: Add N/O-alkylation SMARTS with diverse leaving groups")
        elif "protection" in cand['subclass'].lower():
            lines.append(f"   - Suggestion: Add protection SMARTS "
                        f"(Boc-ON, Cbz-Cl, TBSCl + base patterns)")
        else:
            lines.append(f"   - Suggestion: Analyze example reactions to design appropriate SMARTS")
        lines.append("")

    lines.append("### Strategic Considerations")
    lines.append("")
    lines.append("1. **Deprotections and reductions are the largest gap**: Together they represent "
                f"~{sum(c['count'] for c in expansion_candidates if any(kw in c['subclass'].lower() for kw in ['deprotect', 'reduction']))} "
                "reactions. However, these are often uni-molecular (deprotection) or require specific "
                "reagents (reduction agents like H2/Pd, NaBH4) that may not map cleanly to building block-based templates.")
    lines.append("")
    lines.append("2. **C-C cross-coupling is high-value**: Suzuki, Buchwald-Hartwig, and other "
                "metal-catalyzed couplings are the backbone of medicinal chemistry. Adding these "
                "would dramatically improve the relevance of template-guided exploration.")
    lines.append("")
    lines.append("3. **The model-based approach (T5v2/AIO) naturally covers all these types**: "
                "Template expansion is most useful for: (a) high-confidence reaction generation "
                "within known chemistry, and (b) complementing model predictions with guaranteed-valid products.")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("USPTO 50K Uncovered Reactions Analysis")
    print("=" * 70)

    # 1. Load data
    df = load_all_reactions()
    templates = load_templates()

    # 2. Evaluate coverage
    # Use parallel if many CPUs available, serial otherwise
    num_workers = min(os.cpu_count() or 4, 32)
    print(f"\nEvaluating template coverage ({num_workers} workers)...")

    if num_workers > 1:
        df = evaluate_coverage_parallel(df, templates, num_workers=num_workers)
    else:
        df = evaluate_coverage_batch(df, templates)

    n_covered = df["covered"].sum()
    print(f"\nCoverage: {n_covered}/{len(df)} ({n_covered/len(df)*100:.1f}%)")

    # 3. Generate report
    print("\nGenerating analysis report...")
    report = generate_report(df, templates)

    # 4. Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport written to: {OUTPUT_PATH}")

    # Quick summary
    uncov = df[~df["covered"]]
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(uncov)} uncovered reactions across {len(uncov['class'].unique())} classes")
    print(f"{'='*70}")
    for cls in sorted(uncov["class"].unique()):
        cls_n = (uncov["class"] == cls).sum()
        name = SCHNEIDER_CLASSES.get(cls, "?")
        print(f"  Class {cls:2d} ({name:40s}): {cls_n:5d} uncovered")


if __name__ == "__main__":
    main()
