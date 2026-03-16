"""3D-aware conformer generation and action filtering for MolDQN.

Integrates:
1. Incremental conformer generation (ConstrainedEmbed for large molecules)
2. 3D distance filter (reject bond additions between distant atoms)
3. Bond angle feasibility check
4. Steric clash detection for atom additions

Usage in DQN loop:
    mgr = ConformerManager()
    mgr.set_parent(state_mol)
    filtered, indices = mgr.filter_actions_3d(valid_actions)
    conformers = [mgr.generate_conformer(mol) for mol in filtered]
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

# Reference bond lengths (Å) for single bonds
_BOND_LENGTHS = {
    ('C', 'C'): 1.54, ('C', 'N'): 1.47, ('C', 'O'): 1.43,
    ('N', 'N'): 1.45, ('N', 'O'): 1.40, ('O', 'O'): 1.48,
}

# VDW radii (Å)
_VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'H': 1.20, 'S': 1.80, 'F': 1.47}

# Ideal bond angles by hybridization (degrees)
_IDEAL_ANGLES = {
    Chem.rdchem.HybridizationType.SP3: 109.5,
    Chem.rdchem.HybridizationType.SP2: 120.0,
    Chem.rdchem.HybridizationType.SP: 180.0,
}


def _bond_length(sym1, sym2):
    pair = tuple(sorted([sym1, sym2]))
    return _BOND_LENGTHS.get(pair, 1.54)


def _vdw_radius(sym):
    return _VDW_RADII.get(sym, 1.70)


class ConformerManager:
    """Manages 3D conformers across MolDQN steps.

    Maintains a parent conformer and provides:
    - Incremental conformer generation (ConstrainedEmbed + fallback)
    - 3D action space filtering (distance, angle, steric)
    """

    def __init__(self, atom_threshold=10, distance_factor=2.5,
                 angle_tolerance=30.0, vdw_clash_factor=0.65):
        """
        Args:
            atom_threshold: use ConstrainedEmbed for molecules >= this size
            distance_factor: max bond distance = factor * reference_bond_length
            angle_tolerance: max deviation from ideal bond angle (degrees)
            vdw_clash_factor: clash if dist < factor * sum_of_vdw_radii
        """
        self.atom_threshold = atom_threshold
        self.distance_factor = distance_factor
        self.angle_tolerance = angle_tolerance
        self.vdw_clash_factor = vdw_clash_factor

        self.parent_mol = None       # heavy-atom Mol with conformer
        self.parent_coords = None    # (N, 3) array
        self.parent_mol_h = None     # with-H Mol with conformer (for ConstrainedEmbed)

    # ------------------------------------------------------------------
    # Parent management
    # ------------------------------------------------------------------

    def set_parent(self, mol):
        """Set parent molecule and generate its conformer."""
        self.parent_mol = Chem.RWMol(mol)
        # Generate heavy-atom conformer
        res = AllChem.EmbedMolecule(self.parent_mol, randomSeed=42)
        if res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(self.parent_mol)
            except Exception:
                pass
            self.parent_coords = self.parent_mol.GetConformer().GetPositions()
        else:
            self.parent_coords = None

        # Also keep an H-containing version for inner_smi2coords compatibility
        self.parent_mol_h = AllChem.AddHs(Chem.RWMol(mol))
        res_h = AllChem.EmbedMolecule(self.parent_mol_h, randomSeed=42)
        if res_h == 0:
            try:
                AllChem.MMFFOptimizeMolecule(self.parent_mol_h)
            except Exception:
                pass

    def update_parent_from_action(self, mol, conformer_result):
        """Update parent using a child's already-computed conformer.

        Called after action selection to avoid redundant conformer generation.
        conformer_result: (atoms, coords) from generate_conformer, or None.
        """
        self.parent_mol = Chem.RWMol(mol)
        if conformer_result is not None:
            atoms, coords = conformer_result
            # Rebuild a mol with conformer from the coordinates
            # Extract heavy-atom coords
            heavy_idx = [i for i, a in enumerate(atoms) if a != 'H']
            if heavy_idx and self.parent_mol.GetNumAtoms() == len(heavy_idx):
                conf = Chem.Conformer(len(heavy_idx))
                for new_i, old_i in enumerate(heavy_idx):
                    conf.SetAtomPosition(new_i, Point3D(*coords[old_i].tolist()))
                self.parent_mol.RemoveAllConformers()
                self.parent_mol.AddConformer(conf, assignId=True)
                self.parent_coords = np.array(coords[heavy_idx])
            else:
                # Mismatch, regenerate
                self.set_parent(mol)
        else:
            self.set_parent(mol)

    # ------------------------------------------------------------------
    # 3D action filtering
    # ------------------------------------------------------------------

    def filter_actions_3d(self, child_mols):
        """Filter valid actions based on 3D feasibility.

        Applies: distance filter, bond angle check, steric clash detection.
        Always keeps the last action (no-modification).

        Args:
            child_mols: list of RDKit Mol (valid actions from cenv)

        Returns:
            (filtered_mols, kept_indices, stats_dict)
        """
        if self.parent_coords is None or self.parent_mol is None:
            return child_mols, list(range(len(child_mols))), {'skipped': True}

        n_parent = self.parent_mol.GetNumAtoms()
        filtered = []
        indices = []
        stats = {'total': len(child_mols), 'kept': 0,
                 'dist_reject': 0, 'angle_reject': 0, 'steric_reject': 0}

        for i, child in enumerate(child_mols):
            # Always keep last action (no-modification)
            if i == len(child_mols) - 1:
                filtered.append(child)
                indices.append(i)
                stats['kept'] += 1
                continue

            n_child = child.GetNumAtoms()
            reject_reason = None

            if n_child == n_parent:
                # Same atom count → bond addition/removal/order change
                reject_reason = self._check_bond_edit(child)
            elif n_child == n_parent + 1:
                # Atom addition
                reject_reason = self._check_atom_addition(child)
            # n_child < n_parent: removal → always keep

            if reject_reason is None:
                filtered.append(child)
                indices.append(i)
                stats['kept'] += 1
            else:
                stats[reject_reason] += 1

        return filtered, indices, stats

    def _check_bond_edit(self, child):
        """Check bond addition between existing atoms. Returns reject reason or None."""
        match = child.GetSubstructMatch(self.parent_mol)
        if not match:
            # Try reverse: parent might be a supergraph (bond removal)
            return None  # can't determine, keep

        # Build inverse mapping: child_idx → parent_idx
        inv = {child_idx: parent_idx for parent_idx, child_idx in enumerate(match)}

        # Collect parent bonds
        parent_bonds = set()
        for bond in self.parent_mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            parent_bonds.add((min(a, b), max(a, b)))

        # Find new bonds
        for bond in child.GetBonds():
            ci, cj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if ci in inv and cj in inv:
                pi, pj = inv[ci], inv[cj]
                pair = (min(pi, pj), max(pi, pj))
                if pair not in parent_bonds:
                    # New bond found — apply 3D checks
                    sym_i = self.parent_mol.GetAtomWithIdx(pi).GetSymbol()
                    sym_j = self.parent_mol.GetAtomWithIdx(pj).GetSymbol()

                    # Distance check
                    dist = np.linalg.norm(
                        self.parent_coords[pi] - self.parent_coords[pj]
                    )
                    max_dist = _bond_length(sym_i, sym_j) * self.distance_factor
                    if dist > max_dist:
                        return 'dist_reject'

                    # Bond angle check at both endpoints
                    for center, other in [(pi, pj), (pj, pi)]:
                        reason = self._check_angle_at(center, other)
                        if reason:
                            return reason

        return None

    def _check_atom_addition(self, child):
        """Check atom addition for steric feasibility. Returns reject reason or None."""
        match = child.GetSubstructMatch(self.parent_mol)
        if not match:
            return None

        matched_set = set(match)
        new_atoms = [i for i in range(child.GetNumAtoms()) if i not in matched_set]
        if not new_atoms:
            return None

        inv = {child_idx: parent_idx for parent_idx, child_idx in enumerate(match)}
        new_idx = new_atoms[0]
        new_sym = child.GetAtomWithIdx(new_idx).GetSymbol()
        new_vdw = _vdw_radius(new_sym)

        # Find which parent atom the new atom bonds to
        for neighbor in child.GetAtomWithIdx(new_idx).GetNeighbors():
            ni = neighbor.GetIdx()
            if ni not in inv:
                continue
            parent_ni = inv[ni]
            partner_pos = self.parent_coords[parent_ni]
            bond_len = _bond_length(
                self.parent_mol.GetAtomWithIdx(parent_ni).GetSymbol(), new_sym
            )

            # Check: can we place the new atom at ~bond_len from partner
            # without clashing with nearby atoms?
            partner_neighbors = [
                n.GetIdx() for n in self.parent_mol.GetAtomWithIdx(parent_ni).GetNeighbors()
            ]
            for pi in range(self.parent_mol.GetNumAtoms()):
                if pi == parent_ni or pi in partner_neighbors:
                    continue
                other_pos = self.parent_coords[pi]
                other_sym = self.parent_mol.GetAtomWithIdx(pi).GetSymbol()
                dist_to_partner = np.linalg.norm(other_pos - partner_pos)

                # Minimum possible distance from new atom to this atom:
                # new atom is at bond_len from partner, best case is the
                # triangle inequality: dist >= |dist_to_partner - bond_len|
                min_possible = abs(dist_to_partner - bond_len)
                clash_dist = self.vdw_clash_factor * (new_vdw + _vdw_radius(other_sym))

                if min_possible < clash_dist and dist_to_partner < bond_len * 1.5:
                    return 'steric_reject'

        return None

    def _check_angle_at(self, center_idx, new_neighbor_idx):
        """Check if adding a bond at center_idx creates unreasonable angles."""
        atom = self.parent_mol.GetAtomWithIdx(center_idx)
        hyb = atom.GetHybridization()
        ideal = _IDEAL_ANGLES.get(hyb, 109.5)

        pos_center = self.parent_coords[center_idx]
        pos_new = self.parent_coords[new_neighbor_idx]
        v_new = pos_new - pos_center
        norm_new = np.linalg.norm(v_new)
        if norm_new < 1e-8:
            return None

        for neighbor in atom.GetNeighbors():
            ni = neighbor.GetIdx()
            pos_n = self.parent_coords[ni]
            v_n = pos_n - pos_center
            norm_n = np.linalg.norm(v_n)
            if norm_n < 1e-8:
                continue

            cos_angle = np.dot(v_n, v_new) / (norm_n * norm_new)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            # Reject extremely strained angles
            if angle < 50.0:
                return 'angle_reject'

        return None

    # ------------------------------------------------------------------
    # Conformer generation
    # ------------------------------------------------------------------

    def generate_conformer(self, mol):
        """Generate conformer for a molecule.

        Uses ConstrainedEmbed for large molecules (>= atom_threshold),
        falls back to scratch ETKDG for small molecules or on failure.

        Args:
            mol: RDKit Mol (heavy atoms only)

        Returns:
            (atoms_list, coords_array) including H atoms, or None on failure.
            atoms_list: list of element symbols
            coords_array: np.ndarray (N_with_H, 3) float32
        """
        n = mol.GetNumAtoms()

        if n >= self.atom_threshold and self.parent_mol is not None:
            result = self._embed_constrained_3stage(mol)
            if result is not None:
                return result

        return self._embed_scratch(mol)

    def _embed_scratch(self, mol):
        """Standard ETKDG from scratch, returns (atoms, coords) with H."""
        mol_h = AllChem.AddHs(Chem.RWMol(mol))
        res = AllChem.EmbedMolecule(mol_h, randomSeed=42)
        if res != 0:
            # Fallback to 2D
            AllChem.Compute2DCoords(mol_h)
        else:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h)
            except Exception:
                pass

        atoms = [a.GetSymbol() for a in mol_h.GetAtoms()]
        coords = mol_h.GetConformer().GetPositions().astype(np.float32)
        return atoms, coords

    def _embed_constrained_3stage(self, child_mol):
        """3-stage ConstrainedEmbed with fallback.

        Stage 1: Forward match (parent ⊂ child) → ConstrainedEmbed
        Stage 2: Reverse match (child ⊂ parent) → coordMap from parent
        Stage 3: Scratch ETKDG
        """
        # Stage 1: Forward — parent is substructure of child
        if self.parent_mol.GetNumConformers() > 0:
            match = child_mol.GetSubstructMatch(self.parent_mol)
            if match:
                try:
                    embedded = AllChem.ConstrainedEmbed(
                        Chem.RWMol(child_mol), self.parent_mol, randomseed=42
                    )
                    return self._add_h_coords(embedded)
                except Exception:
                    pass

        # Stage 2: Reverse — child is substructure of parent
        if self.parent_mol.GetNumConformers() > 0:
            match_rev = self.parent_mol.GetSubstructMatch(child_mol)
            if match_rev:
                try:
                    return self._embed_reverse_coordmap(child_mol, match_rev)
                except Exception:
                    pass

        # Stage 3 handled by caller (_embed_scratch)
        return None

    def _embed_reverse_coordmap(self, child_mol, match_in_parent):
        """Embed child using coordinates extracted from parent.

        match_in_parent: result of parent.GetSubstructMatch(child),
                         maps child_atom_idx → parent_atom_idx.
        """
        child_h = AllChem.AddHs(Chem.RWMol(child_mol))
        parent_conf = self.parent_mol.GetConformer()

        # Build coordMap for heavy atoms: child_h_idx → Point3D
        coord_map = {}
        # Map child heavy atoms to child_h indices (heavy atoms keep same order)
        n_child_heavy = child_mol.GetNumAtoms()
        for child_idx in range(n_child_heavy):
            parent_idx = match_in_parent[child_idx]
            pos = parent_conf.GetAtomPosition(parent_idx)
            coord_map[child_idx] = pos

        res = AllChem.EmbedMolecule(child_h, coordMap=coord_map, randomSeed=42)
        if res != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(child_h)
        except Exception:
            pass

        atoms = [a.GetSymbol() for a in child_h.GetAtoms()]
        coords = child_h.GetConformer().GetPositions().astype(np.float32)
        return atoms, coords

    def _add_h_coords(self, mol_heavy):
        """Add H atoms and compute their coordinates.

        mol_heavy: Mol with heavy-atom conformer from ConstrainedEmbed.
        """
        conf_heavy = mol_heavy.GetConformer()
        n_heavy = mol_heavy.GetNumAtoms()

        # Add H atoms
        mol_h = AllChem.AddHs(Chem.RWMol(mol_heavy))

        # Build coordMap pinning heavy atoms
        coord_map = {}
        for i in range(n_heavy):
            coord_map[i] = conf_heavy.GetAtomPosition(i)

        # Embed with heavy atoms fixed, only placing H
        res = AllChem.EmbedMolecule(mol_h, coordMap=coord_map, randomSeed=42)
        if res != 0:
            # Fallback: just use heavy atom coords, set H to (0,0,0)
            # and hope MMFF can fix it
            conf_new = Chem.Conformer(mol_h.GetNumAtoms())
            for i in range(n_heavy):
                conf_new.SetAtomPosition(i, conf_heavy.GetAtomPosition(i))
            for i in range(n_heavy, mol_h.GetNumAtoms()):
                conf_new.SetAtomPosition(i, Point3D(0, 0, 0))
            mol_h.RemoveAllConformers()
            mol_h.AddConformer(conf_new, assignId=True)
            try:
                AllChem.MMFFOptimizeMolecule(mol_h)
            except Exception:
                pass

        else:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h)
            except Exception:
                pass

        atoms = [a.GetSymbol() for a in mol_h.GetAtoms()]
        coords = mol_h.GetConformer().GetPositions().astype(np.float32)
        return atoms, coords
