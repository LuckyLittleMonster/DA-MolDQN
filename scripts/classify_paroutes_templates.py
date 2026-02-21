#!/usr/bin/env python3
"""Classify PaRoutes reactions into AiZynth template codes.

Run in the rdchiral conda environment:
    conda run -n rdchiral python scripts/classify_paroutes_templates.py

Strategy:
1. rdchiral extraction + direct string match (~52%)
2. Functional matching (apply all AiZynth retro templates to product) for remainder
3. Multiprocessing for functional matching to use all available cores

Outputs:
- Data/paroutes/n1_routes_annotated.json  — routes with template_code injected (all routes)
- Data/paroutes/n1_routes_matched.json    — only routes where ALL reactions matched
- Data/paroutes/n1_unmatched_reactions.json — unmatched reactions detail
- Data/paroutes/n1_classification_report.txt — statistics
- template/data/templates_paroutes.txt — forward SMARTS for matched templates
- template/data/paroutes_code_to_idx.pkl — template_code → file line index
- template/data/building_blocks_paroutes.smi — PaRoutes BBs (canonical SMILES)
"""

import csv
import gzip
import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'Data' / 'paroutes'
TEMPLATE_DIR = PROJECT_ROOT / 'rl' / 'template' / 'data'
AIZYNTH_TEMPLATES = PROJECT_ROOT / 'refs' / 'aizynthfinder' / 'data' / 'uspto_templates.csv.gz'


# ---------------------------------------------------------------------------
# Template library loading
# ---------------------------------------------------------------------------

def load_aizynth_templates():
    """Load AiZynth template library.

    Returns:
        retro_to_code: dict mapping retro_template string → template_code
        code_to_retro: dict mapping template_code → retro_template string
        compiled_retro: list of (template_code, compiled RDKit reaction)
    """
    retro_to_code = {}
    code_to_retro = {}
    compiled_retro = []

    with gzip.open(str(AIZYNTH_TEMPLATES), 'rt') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            code = int(row['template_code'])
            retro = row['retro_template']
            retro_to_code[retro] = code
            code_to_retro[code] = retro

            try:
                rxn = AllChem.ReactionFromSmarts(retro)
                if rxn is not None:
                    compiled_retro.append((code, rxn))
            except Exception:
                pass

    return retro_to_code, code_to_retro, compiled_retro


# ---------------------------------------------------------------------------
# Tree traversal — collect reactions with product/reactant SMILES
# ---------------------------------------------------------------------------

def collect_reactions(node, depth=0):
    """Walk a PaRoutes route tree and yield reaction info dicts."""
    if node['type'] == 'mol' and 'children' in node and node['children']:
        rxn_node = node['children'][0]
        if rxn_node['type'] == 'reaction':
            meta = rxn_node.get('metadata', {})
            child_mols = [c for c in rxn_node.get('children', [])
                          if c.get('type') == 'mol']
            yield {
                'product_smi': node['smiles'],
                'reactant_smis': [c['smiles'] for c in child_mols],
                'in_stock': [c.get('in_stock', False) for c in child_mols],
                'metadata': meta,
                'depth': depth,
                'ringbreaker': meta.get('RingBreaker', False),
            }
            for c in child_mols:
                yield from collect_reactions(c, depth + 1)


# ---------------------------------------------------------------------------
# rdchiral extraction matching
# ---------------------------------------------------------------------------

def try_rdchiral_match(rxn_info, retro_to_code):
    """Try to match via rdchiral template extraction + string lookup.

    Returns template_code or None.
    """
    try:
        from rdchiral.template_extractor import extract_from_reaction
    except ImportError:
        return None

    rxn_smi = rxn_info['metadata'].get('smiles', '')
    if '>>' not in rxn_smi:
        return None

    parts = rxn_smi.split('>>')
    rxn_dict = {'_id': '0', 'reactants': parts[0], 'products': parts[1]}

    try:
        result = extract_from_reaction(rxn_dict)
        if 'reaction_smarts' in result:
            retro = result['reaction_smarts']
            if retro in retro_to_code:
                return retro_to_code[retro]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Functional matching (apply all templates to product)
# ---------------------------------------------------------------------------

# Global variable for worker processes
_compiled_retro = None


def _init_worker(templates_data):
    """Initialize worker with compiled templates."""
    global _compiled_retro
    _compiled_retro = []
    for code, retro_smarts in templates_data:
        try:
            rxn = AllChem.ReactionFromSmarts(retro_smarts)
            if rxn is not None:
                _compiled_retro.append((code, rxn))
        except Exception:
            pass


def _functional_match_worker(args):
    """Worker function for functional matching.

    Args:
        args: (product_smi, target_reactant_smis_canonical)

    Returns:
        template_code or -1
    """
    product_smi, target_reactants_set = args
    product_mol = Chem.MolFromSmiles(product_smi)
    if product_mol is None:
        return -1

    for code, retro_rxn in _compiled_retro:
        try:
            outcomes = retro_rxn.RunReactants((product_mol,))
            for outcome in outcomes:
                got = set()
                for mol in outcome:
                    try:
                        Chem.SanitizeMol(mol)
                        got.add(Chem.MolToSmiles(mol))
                    except Exception:
                        pass
                if got == target_reactants_set:
                    return code
        except Exception:
            pass
    return -1


def functional_match_batch(rxn_infos, code_to_retro, n_workers=None):
    """Functional matching for a batch of reactions using multiprocessing.

    Returns:
        list of template_codes (or -1 for unmatched)
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 32)

    # Prepare canonical reactant sets
    tasks = []
    for info in rxn_infos:
        product_smi = info['product_smi']
        target = set()
        for s in info['reactant_smis']:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                target.add(Chem.MolToSmiles(mol))
        tasks.append((product_smi, target))

    # Prepare template data for workers (code, retro_smarts)
    templates_data = [(code, retro) for code, retro in code_to_retro.items()]

    print(f'  Functional matching {len(tasks)} reactions with {n_workers} workers...')
    t0 = time.time()

    with Pool(n_workers, initializer=_init_worker,
              initargs=(templates_data,)) as pool:
        results = pool.map(_functional_match_worker, tasks, chunksize=4)

    t1 = time.time()
    n_matched = sum(1 for r in results if r >= 0)
    print(f'  Done in {t1 - t0:.1f}s, matched {n_matched}/{len(tasks)}')
    return results


# ---------------------------------------------------------------------------
# Forward SMARTS generation
# ---------------------------------------------------------------------------

def reverse_smarts(smarts):
    """Swap reactants and products in a SMARTS string (retro → forward)."""
    parts = smarts.split('>')
    if len(parts) == 3:
        return f"{parts[2]}>{parts[1]}>{parts[0]}"
    elif len(parts) == 2:
        return f"{parts[1]}>>{parts[0]}"
    raise ValueError(f"Cannot parse SMARTS: {smarts}")


# ---------------------------------------------------------------------------
# BB file generation
# ---------------------------------------------------------------------------

def generate_bb_file(stock_path, output_path):
    """Convert PaRoutes stock file to canonical SMILES BB file."""
    canonical = []
    with open(stock_path) as f:
        for line in f:
            smi = line.strip()
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical.append(Chem.MolToSmiles(mol))
            else:
                canonical.append(smi)  # keep original if parse fails

    with open(output_path, 'w') as f:
        for smi in canonical:
            f.write(smi + '\n')

    return len(canonical)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # --- Load AiZynth templates ---
    print('Loading AiZynth templates...')
    retro_to_code, code_to_retro, compiled_retro = load_aizynth_templates()
    print(f'  {len(retro_to_code)} templates loaded, {len(compiled_retro)} compiled')

    # --- Load PaRoutes ---
    routes_path = DATA_DIR / 'n1_routes.json'
    print(f'Loading PaRoutes from {routes_path}...')
    with open(routes_path) as f:
        routes = json.load(f)
    print(f'  {len(routes)} routes loaded')

    # --- Collect all reactions ---
    print('Collecting reactions from route trees...')
    all_reactions = []
    for route_idx, route in enumerate(routes):
        for rxn_info in collect_reactions(route):
            rxn_info['route_idx'] = route_idx
            rxn_info['target_smi'] = route['smiles']
            all_reactions.append(rxn_info)
    print(f'  {len(all_reactions)} total reactions')

    # --- Phase 1: rdchiral string matching ---
    print('\nPhase 1: rdchiral template extraction + string matching...')
    t1 = time.time()
    phase1_matched = 0
    phase1_codes = {}  # index → template_code
    for i, rxn in enumerate(all_reactions):
        code = try_rdchiral_match(rxn, retro_to_code)
        if code is not None:
            phase1_codes[i] = code
            phase1_matched += 1
    t2 = time.time()
    print(f'  Phase 1: {phase1_matched}/{len(all_reactions)} matched '
          f'({phase1_matched / len(all_reactions) * 100:.1f}%) in {t2 - t1:.1f}s')

    # --- Phase 2: functional matching for remaining ---
    unmatched_indices = [i for i in range(len(all_reactions))
                         if i not in phase1_codes]
    unmatched_rxns = [all_reactions[i] for i in unmatched_indices]

    phase2_codes = {}
    if unmatched_rxns:
        print(f'\nPhase 2: Functional matching for {len(unmatched_rxns)} '
              f'remaining reactions...')
        results = functional_match_batch(unmatched_rxns, code_to_retro)
        for idx, code in zip(unmatched_indices, results):
            if code >= 0:
                phase2_codes[idx] = code

    total_matched = len(phase1_codes) + len(phase2_codes)
    print(f'\nTotal matched: {total_matched}/{len(all_reactions)} '
          f'({total_matched / len(all_reactions) * 100:.1f}%)')

    # --- Merge results and annotate routes ---
    all_codes = {**phase1_codes, **phase2_codes}

    # Build reaction index → template_code map
    rxn_code_map = {}
    for i, rxn in enumerate(all_reactions):
        code = all_codes.get(i, -1)
        rxn_code_map[i] = code

    # Annotate route trees
    print('\nAnnotating route trees...')
    annotated_routes = json.loads(json.dumps(routes))  # deep copy

    rxn_counter = [0]  # mutable counter for closure

    def annotate_tree(node):
        if node['type'] == 'mol' and 'children' in node and node['children']:
            rxn_node = node['children'][0]
            if rxn_node['type'] == 'reaction':
                idx = rxn_counter[0]
                code = rxn_code_map.get(idx, -1)
                if 'metadata' not in rxn_node:
                    rxn_node['metadata'] = {}
                rxn_node['metadata']['template_code'] = code
                rxn_counter[0] += 1

                for c in rxn_node.get('children', []):
                    annotate_tree(c)

    for route in annotated_routes:
        annotate_tree(route)

    # Save annotated routes (all, including unmatched)
    out_annotated = DATA_DIR / 'n1_routes_annotated.json'
    with open(out_annotated, 'w') as f:
        json.dump(annotated_routes, f)
    print(f'  Saved annotated routes to {out_annotated}')

    # Filter: keep only routes where ALL reactions have template_code != -1
    def route_fully_matched(node):
        """Check if ALL reactions in this route tree have template_code != -1."""
        if node['type'] == 'mol' and 'children' in node and node['children']:
            rxn = node['children'][0]
            if rxn['type'] == 'reaction':
                code = rxn.get('metadata', {}).get('template_code', -1)
                if code == -1:
                    return False
                for c in rxn.get('children', []):
                    if not route_fully_matched(c):
                        return False
        return True

    matched_routes = [r for r in annotated_routes if route_fully_matched(r)]
    out_matched = DATA_DIR / 'n1_routes_matched.json'
    with open(out_matched, 'w') as f:
        json.dump(matched_routes, f)
    print(f'  Saved {len(matched_routes)}/{len(annotated_routes)} fully-matched '
          f'routes to {out_matched}')

    # --- Unmatched reactions report ---
    unmatched_info = []
    for i, rxn in enumerate(all_reactions):
        if i not in all_codes:
            info = {
                'route_idx': rxn['route_idx'],
                'target_smi': rxn['target_smi'],
                'depth': rxn['depth'],
                'product_smi': rxn['product_smi'],
                'reactant_smis': rxn['reactant_smis'],
                'ringbreaker': rxn['ringbreaker'],
                'reason': 'ringbreaker' if rxn['ringbreaker'] else 'no_template_match',
            }
            # Try to get rdchiral extraction info
            rxn_smi = rxn['metadata'].get('smiles', '')
            if '>>' in rxn_smi:
                try:
                    from rdchiral.template_extractor import extract_from_reaction
                    parts = rxn_smi.split('>>')
                    result = extract_from_reaction({
                        '_id': '0', 'reactants': parts[0], 'products': parts[1]
                    })
                    if 'reaction_smarts' in result:
                        info['extracted_retro_template'] = result['reaction_smarts']
                    else:
                        info['extraction_error'] = str(result)
                except Exception as e:
                    info['extraction_error'] = str(e)
            unmatched_info.append(info)

    out_unmatched = DATA_DIR / 'n1_unmatched_reactions.json'
    with open(out_unmatched, 'w') as f:
        json.dump(unmatched_info, f, indent=2)
    print(f'  Saved {len(unmatched_info)} unmatched reactions to {out_unmatched}')

    # --- Statistics ---
    unique_codes = set(all_codes.values())
    ringbreaker_count = sum(1 for rxn in all_reactions if rxn['ringbreaker'])
    depth_counter = Counter()
    unmatched_depth = Counter()
    for i, rxn in enumerate(all_reactions):
        depth_counter[rxn['depth']] += 1
        if i not in all_codes:
            unmatched_depth[rxn['depth']] += 1

    report_lines = [
        f'PaRoutes n1 Template Classification Report',
        f'==========================================',
        f'',
        f'Total routes: {len(routes)}',
        f'Total reactions: {len(all_reactions)}',
        f'',
        f'Phase 1 (rdchiral string match): {phase1_matched} ({phase1_matched / len(all_reactions) * 100:.1f}%)',
        f'Phase 2 (functional match): {len(phase2_codes)} ({len(phase2_codes) / len(all_reactions) * 100:.1f}%)',
        f'Total matched: {total_matched} ({total_matched / len(all_reactions) * 100:.1f}%)',
        f'Unmatched: {len(unmatched_info)} ({len(unmatched_info) / len(all_reactions) * 100:.1f}%)',
        f'  RingBreaker reactions: {ringbreaker_count}',
        f'',
        f'Unique template_codes used: {len(unique_codes)}',
        f'',
        f'Fully-matched routes: {len(matched_routes)}/{len(routes)} '
        f'({len(matched_routes) / len(routes) * 100:.1f}%)',
        f'Routes with unmatched reactions: {len(routes) - len(matched_routes)}',
        f'',
        f'Depth distribution:',
    ]
    for depth in sorted(depth_counter.keys()):
        total = depth_counter[depth]
        um = unmatched_depth.get(depth, 0)
        report_lines.append(
            f'  depth {depth}: {total} reactions, {um} unmatched ({um / total * 100:.1f}%)')

    report_lines.append(f'')
    report_lines.append(f'Time: {time.time() - t_start:.1f}s')

    report = '\n'.join(report_lines)
    print(f'\n{report}')

    out_report = DATA_DIR / 'n1_classification_report.txt'
    with open(out_report, 'w') as f:
        f.write(report)
    print(f'\nSaved report to {out_report}')

    # --- Generate template files ---
    print('\nGenerating template files...')

    # Collect unique template codes and their retro SMARTS
    used_codes = sorted(unique_codes)
    forward_smarts_list = []
    valid_codes = []
    for code in used_codes:
        retro = code_to_retro.get(code)
        if retro is None:
            continue
        try:
            fwd = reverse_smarts(retro)
            forward_smarts_list.append(fwd)
            valid_codes.append(code)
        except ValueError:
            print(f'  Warning: cannot reverse template code={code}')

    # Write templates_paroutes.txt
    out_templates = TEMPLATE_DIR / 'templates_paroutes.txt'
    with open(out_templates, 'w') as f:
        for fwd, code in zip(forward_smarts_list, valid_codes):
            f.write(f'{fwd}  # aizynth_code={code}\n')
    print(f'  Saved {len(valid_codes)} templates to {out_templates}')

    # Write paroutes_code_to_idx.pkl
    code_to_idx = {code: idx for idx, code in enumerate(valid_codes)}
    out_pkl = TEMPLATE_DIR / 'paroutes_code_to_idx.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(code_to_idx, f)
    print(f'  Saved code_to_idx ({len(code_to_idx)} entries) to {out_pkl}')

    # --- Generate BB file ---
    print('\nGenerating building blocks file...')
    stock_path = DATA_DIR / 'n1_stock.txt'
    out_bb = TEMPLATE_DIR / 'building_blocks_paroutes.smi'
    n_bb = generate_bb_file(stock_path, out_bb)
    print(f'  Saved {n_bb} building blocks to {out_bb}')

    print(f'\nAll done! Total time: {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
