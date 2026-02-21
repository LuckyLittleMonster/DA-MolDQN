"""Analyze PaRoutes dataset: route statistics, step distribution, BB coverage, molecular properties."""

import json
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


def count_steps(node, depth=0):
    """Count the number of reaction steps (depth) in a route tree."""
    if node['type'] == 'mol':
        if not node.get('children'):
            return depth  # leaf (BB)
        max_depth = depth
        for child in node['children']:
            d = count_steps(child, depth)
            max_depth = max(max_depth, d)
        return max_depth
    elif node['type'] == 'reaction':
        max_depth = depth + 1
        for child in node.get('children', []):
            d = count_steps(child, depth + 1)
            max_depth = max(max_depth, d)
        return max_depth
    return depth


def count_reactions(node):
    """Count total number of reaction nodes in a route tree."""
    if node['type'] == 'reaction':
        total = 1
        for child in node.get('children', []):
            total += count_reactions(child)
        return total
    elif node['type'] == 'mol':
        total = 0
        for child in node.get('children', []):
            total += count_reactions(child)
        return total
    return 0


def extract_molecules(node, mols=None, bbs=None, intermediates=None):
    """Extract all molecules from a route tree, categorized."""
    if mols is None:
        mols = set()
        bbs = set()
        intermediates = set()

    if node['type'] == 'mol':
        smi = node.get('smiles', '')
        if smi:
            mols.add(smi)
            if node.get('in_stock', False) or not node.get('children'):
                bbs.add(smi)
            elif node.get('children'):
                intermediates.add(smi)
        for child in node.get('children', []):
            extract_molecules(child, mols, bbs, intermediates)

    elif node['type'] == 'reaction':
        for child in node.get('children', []):
            extract_molecules(child, mols, bbs, intermediates)

    return mols, bbs, intermediates


def extract_reactions(node, reactions=None):
    """Extract reaction nodes with their reactant/product info."""
    if reactions is None:
        reactions = []

    if node['type'] == 'reaction':
        children_smiles = []
        for child in node.get('children', []):
            if child['type'] == 'mol':
                children_smiles.append(child.get('smiles', ''))
        # The parent mol node's SMILES is the product
        rxn_info = {
            'reactants': children_smiles,
            'metadata': node.get('metadata', {}),
            'n_reactants': len(children_smiles),
        }
        reactions.append(rxn_info)
        for child in node.get('children', []):
            extract_reactions(child, reactions)

    elif node['type'] == 'mol':
        for child in node.get('children', []):
            extract_reactions(child, reactions)

    return reactions


def compute_mol_props(smiles):
    """Compute basic molecular properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': Descriptors.ExactMolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': rdMolDescriptors.CalcNumHBD(mol),
        'HBA': rdMolDescriptors.CalcNumHBA(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'QED': QED.qed(mol),
        'HeavyAtoms': mol.GetNumHeavyAtoms(),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
    }


def analyze_routes(routes_file, stock_file=None, targets_file=None, name=""):
    """Full analysis of a PaRoutes benchmark set."""
    print(f"\n{'='*70}")
    print(f"  PaRoutes Analysis: {name}")
    print(f"{'='*70}")

    # Load routes
    print(f"\nLoading {routes_file}...")
    with open(routes_file) as f:
        routes = json.load(f)
    print(f"  Total routes: {len(routes)}")

    # Load stock if available
    stock_smiles = set()
    if stock_file and Path(stock_file).exists():
        with open(stock_file) as f:
            stock_smiles = {line.strip() for line in f if line.strip()}
        print(f"  Stock BBs: {len(stock_smiles)}")

    # Load targets if available
    target_smiles = []
    if targets_file and Path(targets_file).exists():
        with open(targets_file) as f:
            target_smiles = [line.strip() for line in f if line.strip()]
        print(f"  Target molecules: {len(target_smiles)}")

    # --- Route Statistics ---
    print(f"\n--- Route Structure ---")
    step_counts = []
    reaction_counts = []
    all_bbs = set()
    all_intermediates = set()
    all_targets = set()
    all_reactions = []
    reactant_count_dist = Counter()

    for route in routes:
        steps = count_steps(route)
        n_rxns = count_reactions(route)
        step_counts.append(steps)
        reaction_counts.append(n_rxns)

        mols, bbs, intermediates = extract_molecules(route)
        all_bbs.update(bbs)
        all_intermediates.update(intermediates)
        if route.get('smiles'):
            all_targets.add(route['smiles'])

        rxns = extract_reactions(route)
        all_reactions.extend(rxns)
        for rxn in rxns:
            reactant_count_dist[rxn['n_reactants']] += 1

    step_counter = Counter(step_counts)
    print(f"\n  Step distribution (longest linear path):")
    for k in sorted(step_counter.keys()):
        pct = step_counter[k] / len(routes) * 100
        bar = '#' * int(pct / 2)
        print(f"    {k} steps: {step_counter[k]:>5} ({pct:5.1f}%) {bar}")

    print(f"\n  Reaction count distribution:")
    rxn_counter = Counter(reaction_counts)
    for k in sorted(rxn_counter.keys()):
        pct = rxn_counter[k] / len(routes) * 100
        bar = '#' * int(pct / 2)
        print(f"    {k} rxns:  {rxn_counter[k]:>5} ({pct:5.1f}%) {bar}")

    avg_steps = sum(step_counts) / len(step_counts)
    avg_rxns = sum(reaction_counts) / len(reaction_counts)
    print(f"\n  Average steps/route: {avg_steps:.2f}")
    print(f"  Average reactions/route: {avg_rxns:.2f}")
    print(f"  Max steps: {max(step_counts)}")

    # Filter routes by step count for RL use
    routes_1_5 = sum(1 for s in step_counts if 1 <= s <= 5)
    routes_2_5 = sum(1 for s in step_counts if 2 <= s <= 5)
    print(f"\n  Routes with 1-5 steps: {routes_1_5} ({routes_1_5/len(routes)*100:.1f}%)")
    print(f"  Routes with 2-5 steps: {routes_2_5} ({routes_2_5/len(routes)*100:.1f}%)")

    # --- Molecule Statistics ---
    print(f"\n--- Molecule Statistics ---")
    print(f"  Unique target molecules: {len(all_targets)}")
    print(f"  Unique building blocks: {len(all_bbs)}")
    print(f"  Unique intermediates: {len(all_intermediates)}")
    print(f"  Total unique molecules: {len(all_bbs | all_intermediates | all_targets)}")

    # BB overlap with stock
    if stock_smiles:
        bb_in_stock = all_bbs & stock_smiles
        print(f"\n  BBs in stock file: {len(bb_in_stock)}/{len(all_bbs)} ({len(bb_in_stock)/max(len(all_bbs),1)*100:.1f}%)")

    # --- Reaction Statistics ---
    print(f"\n--- Reaction Statistics ---")
    print(f"  Total reactions across all routes: {len(all_reactions)}")
    print(f"  Reactant count distribution:")
    for k in sorted(reactant_count_dist.keys()):
        pct = reactant_count_dist[k] / len(all_reactions) * 100
        print(f"    {k} reactants: {reactant_count_dist[k]:>6} ({pct:.1f}%)")

    # --- Molecular Properties (target molecules) ---
    print(f"\n--- Target Molecule Properties ---")
    # Sample up to 1000 targets for property analysis
    sample_targets = list(all_targets)[:1000]
    props_list = []
    for smi in sample_targets:
        p = compute_mol_props(smi)
        if p:
            props_list.append(p)

    if props_list:
        print(f"  Analyzed {len(props_list)} target molecules:")
        for key in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'QED', 'HeavyAtoms']:
            values = [p[key] for p in props_list]
            mean_v = sum(values) / len(values)
            min_v = min(values)
            max_v = max(values)
            median_v = sorted(values)[len(values)//2]
            print(f"    {key:>12}: mean={mean_v:7.1f}, median={median_v:7.1f}, min={min_v:7.1f}, max={max_v:7.1f}")

        # Lipinski compliance
        ro5_pass = sum(1 for p in props_list
                      if p['MW'] < 500 and p['LogP'] < 5 and p['HBD'] <= 5 and p['HBA'] <= 10)
        print(f"\n  Lipinski Ro5 compliance: {ro5_pass}/{len(props_list)} ({ro5_pass/len(props_list)*100:.1f}%)")

        # Drug-like range (QED > 0.5)
        druglike = sum(1 for p in props_list if p['QED'] > 0.5)
        print(f"  QED > 0.5 (drug-like): {druglike}/{len(props_list)} ({druglike/len(props_list)*100:.1f}%)")

    # --- BB Molecular Properties ---
    print(f"\n--- Building Block Properties ---")
    sample_bbs = list(all_bbs)[:2000]
    bb_props = []
    for smi in sample_bbs:
        p = compute_mol_props(smi)
        if p:
            bb_props.append(p)

    if bb_props:
        print(f"  Analyzed {len(bb_props)} building blocks:")
        for key in ['MW', 'LogP', 'HeavyAtoms']:
            values = [p[key] for p in bb_props]
            mean_v = sum(values) / len(values)
            median_v = sorted(values)[len(values)//2]
            print(f"    {key:>12}: mean={mean_v:7.1f}, median={median_v:7.1f}")

    # --- BB overlap with our library ---
    print(f"\n--- BB Overlap with Project Libraries ---")
    our_bb_files = [
        ('ZINC 10K', 'rl/template/data/building_blocks.smi.gz'),
        ('Merged 37K', 'rl/template/data/building_blocks_merged.smi'),
        ('AiZynth 186', 'rl/template/data/building_blocks_aizynth.smi'),
    ]

    project_root = Path(__file__).resolve().parent.parent

    # Canonicalize PaRoutes BBs for fair comparison
    canon_paroutes_bbs = set()
    for smi in all_bbs:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canon_paroutes_bbs.add(Chem.MolToSmiles(mol))

    for name_lib, rel_path in our_bb_files:
        path = project_root / rel_path
        if not path.exists():
            print(f"  {name_lib}: file not found ({path})")
            continue

        our_bbs = set()
        if str(path).endswith('.gz'):
            import gzip as gz
            with gz.open(path, 'rt') as f:
                for line in f:
                    smi = line.strip().split()[0]
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        our_bbs.add(Chem.MolToSmiles(mol))
        else:
            with open(path) as f:
                for line in f:
                    smi = line.strip().split()[0]
                    if smi:
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            our_bbs.add(Chem.MolToSmiles(mol))

        overlap = canon_paroutes_bbs & our_bbs
        print(f"  {name_lib} ({len(our_bbs)} BBs): overlap = {len(overlap)} "
              f"({len(overlap)/max(len(canon_paroutes_bbs),1)*100:.1f}% of PaRoutes BBs)")

    # --- Example Routes ---
    print(f"\n--- Example Routes (first 3) ---")
    for i, route in enumerate(routes[:3]):
        target = route.get('smiles', '?')
        steps = count_steps(route)
        n_rxns = count_reactions(route)
        mols, bbs, inters = extract_molecules(route)
        print(f"\n  Route {i+1}:")
        print(f"    Target: {target[:80]}")
        print(f"    Steps: {steps}, Reactions: {n_rxns}")
        print(f"    BBs ({len(bbs)}): {', '.join(list(bbs)[:3])}{'...' if len(bbs)>3 else ''}")
        props = compute_mol_props(target)
        if props:
            print(f"    MW={props['MW']:.0f}, LogP={props['LogP']:.1f}, QED={props['QED']:.3f}")

    return {
        'n_routes': len(routes),
        'step_counts': step_counts,
        'n_bbs': len(all_bbs),
        'n_targets': len(all_targets),
        'n_reactions': len(all_reactions),
        'routes_1_5': routes_1_5,
        'routes_2_5': routes_2_5,
    }


def main():
    data_dir = Path('/shared/data1/Users/l1062811/git/DA-MolDQN/Data/paroutes')

    # Analyze n1 benchmark
    stats_n1 = analyze_routes(
        data_dir / 'n1_routes.json',
        stock_file=data_dir / 'n1_stock.txt',
        targets_file=data_dir / 'n1_targets.txt',
        name='N1 Benchmark (10K routes)'
    )

    # Analyze n5 benchmark
    stats_n5 = analyze_routes(
        data_dir / 'n5_routes.json',
        stock_file=data_dir / 'n5_stock.txt',
        targets_file=data_dir / 'n5_targets.txt',
        name='N5 Benchmark (10K routes)'
    )

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"  {'':>25} {'N1':>10} {'N5':>10}")
    print(f"  {'Total routes':>25} {stats_n1['n_routes']:>10} {stats_n5['n_routes']:>10}")
    print(f"  {'Unique targets':>25} {stats_n1['n_targets']:>10} {stats_n5['n_targets']:>10}")
    print(f"  {'Unique BBs':>25} {stats_n1['n_bbs']:>10} {stats_n5['n_bbs']:>10}")
    print(f"  {'Total reactions':>25} {stats_n1['n_reactions']:>10} {stats_n5['n_reactions']:>10}")
    print(f"  {'Routes 1-5 steps':>25} {stats_n1['routes_1_5']:>10} {stats_n5['routes_1_5']:>10}")
    print(f"  {'Routes 2-5 steps':>25} {stats_n1['routes_2_5']:>10} {stats_n5['routes_2_5']:>10}")
    avg1 = sum(stats_n1['step_counts']) / len(stats_n1['step_counts'])
    avg5 = sum(stats_n5['step_counts']) / len(stats_n5['step_counts'])
    print(f"  {'Avg steps/route':>25} {avg1:>10.2f} {avg5:>10.2f}")


if __name__ == '__main__':
    main()
