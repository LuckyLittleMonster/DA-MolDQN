"""Reaction template wrappers around RDKit's ChemicalReaction.

Adapted from RxnFlow (refs/RxnFlow/src/rxnflow/envs/reaction.py).
Key change: BiReaction uses `is_mol_first` (DA-MolDQN perspective: current molecule)
instead of RxnFlow's `is_block_first` (building block perspective).
"""

from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts


class Reaction:
    """Base reaction class wrapping a SMARTS template."""

    def __init__(self, template: str, index: int = -1):
        self.template: str = template
        self.index: int = index
        self._rxn_forward: ChemicalReaction = ReactionFromSmarts(template)
        ChemicalReaction.Initialize(self._rxn_forward)
        self.num_reactants: int = self._rxn_forward.GetNumReactantTemplates()
        self.num_products: int = self._rxn_forward.GetNumProductTemplates()

        self.reactant_pattern: list[RDMol] = []
        for i in range(self.num_reactants):
            self.reactant_pattern.append(self._rxn_forward.GetReactantTemplate(i))

    def is_reactant(self, mol: RDMol, order: int | None = None) -> bool:
        """Check if mol matches this reaction's reactant pattern."""
        if order is None:
            return self._rxn_forward.IsMoleculeReactant(mol)
        else:
            return mol.HasSubstructMatch(self.reactant_pattern[order])

    def forward(self, *reactants: RDMol) -> list[tuple[RDMol, ...]]:
        """Run forward reaction, returning deduplicated refined products."""
        assert len(reactants) == self.num_reactants, (
            f"Expected {self.num_reactants} reactants, got {len(reactants)}")
        return _run_reaction(self._rxn_forward, reactants,
                             self.num_reactants, self.num_products)

    def forward_smiles(self, *reactants: RDMol) -> list[str]:
        """Run forward reaction, returning deduplicated product SMILES."""
        assert len(reactants) == self.num_reactants, (
            f"Expected {self.num_reactants} reactants, got {len(reactants)}")
        return _run_reaction_smiles(self._rxn_forward, reactants,
                                    self.num_reactants, self.num_products)


class UniReaction(Reaction):
    """Uni-molecular reaction: 1 reactant -> 1 product."""

    def __init__(self, template: str, index: int = -1):
        super().__init__(template, index)
        assert self.num_reactants == 1, f"UniReaction requires 1 reactant, got {self.num_reactants}"
        assert self.num_products == 1, f"UniReaction requires 1 product, got {self.num_products}"


class BiReaction(Reaction):
    """Bi-molecular reaction: current_mol + building_block -> product.

    `is_mol_first`: True if the current molecule is reactant[0] in the SMARTS,
                    False if it's reactant[1].
    """

    def __init__(self, template: str, is_mol_first: bool, index: int = -1):
        super().__init__(template, index)
        self.is_mol_first: bool = is_mol_first
        # mol_order: which reactant position the current molecule occupies
        self.mol_order: int = 0 if is_mol_first else 1
        # block_order: which reactant position the building block occupies
        self.block_order: int = 1 if is_mol_first else 0
        assert self.num_reactants == 2, f"BiReaction requires 2 reactants, got {self.num_reactants}"
        assert self.num_products == 1, f"BiReaction requires 1 product, got {self.num_products}"

    def is_mol_reactant(self, mol: RDMol) -> bool:
        """Check if mol can serve as the 'current molecule' in this reaction."""
        return mol.HasSubstructMatch(self.reactant_pattern[self.mol_order])

    def is_block_reactant(self, block: RDMol) -> bool:
        """Check if block can serve as the building block in this reaction."""
        return block.HasSubstructMatch(self.reactant_pattern[self.block_order])

    def forward(self, mol: RDMol, block: RDMol) -> list[tuple[RDMol, ...]]:
        """Run reaction with mol as current molecule and block as building block."""
        if self.is_mol_first:
            reactants = (mol, block)
        else:
            reactants = (block, mol)
        return _run_reaction(self._rxn_forward, reactants,
                             self.num_reactants, self.num_products)

    def forward_smiles(self, mol: RDMol, block: RDMol) -> list[str]:
        """Run reaction returning product SMILES."""
        if self.is_mol_first:
            reactants = (mol, block)
        else:
            reactants = (block, mol)
        return _run_reaction_smiles(self._rxn_forward, reactants,
                                    self.num_reactants, self.num_products)


def _run_reaction(
    reaction: ChemicalReaction,
    reactants: tuple[RDMol, ...],
    num_reactants: int,
    num_products: int,
) -> list[tuple[RDMol, ...]]:
    """Run RDKit RunReactants with dedup and sanitization."""
    assert len(reactants) == num_reactants
    ps = reaction.RunReactants(reactants, 5)

    refine_ps = []
    for p in ps:
        if len(p) != num_products:
            continue
        refined = []
        for mol in p:
            mol = _refine_mol(mol)
            if mol is None:
                break
            refined.append(mol)
        if len(refined) == num_products:
            refine_ps.append(tuple(refined))

    # Deduplicate by canonical SMILES
    unique_ps = []
    seen = set()
    for p in refine_ps:
        key = tuple(Chem.MolToSmiles(mol) for mol in p)
        if key not in seen:
            seen.add(key)
            unique_ps.append(p)
    return unique_ps


def _run_reaction_smiles(
    reaction: ChemicalReaction,
    reactants: tuple[RDMol, ...],
    num_reactants: int,
    num_products: int,
) -> list[str]:
    """Run reaction returning unique product SMILES (for single-product reactions)."""
    assert len(reactants) == num_reactants
    ps = reaction.RunReactants(reactants, 5)

    results = set()
    for p in ps:
        if len(p) != num_products:
            continue
        smiles_list = []
        for mol in p:
            try:
                mol = Chem.RemoveHs(mol, updateExplicitCount=True)
                smi = Chem.MolToSmiles(mol)
            except Exception:
                break
            smi = smi.replace("[C]", "C").replace("[N]", "N").replace("[CH]", "C")
            smiles_list.append(smi)
        if len(smiles_list) == num_products:
            results.update(smiles_list)
    return list(results)


def _refine_mol(mol: RDMol) -> RDMol | None:
    """Sanitize and canonicalize a product molecule."""
    try:
        mol = Chem.RemoveHs(mol, updateExplicitCount=True)
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi, replacements={"[C]": "C", "[N]": "N", "[CH]": "C"})
    except Exception:
        return None
    return mol


def load_templates(path: str) -> tuple[list[UniReaction], list[BiReaction]]:
    """Load reaction templates from a text file.

    Each line is a SMARTS reaction template. Templates with 1 reactant
    become UniReactions; templates with 2 reactants become BiReactions
    (creating one BiReaction per possible molecule position).

    Returns (uni_reactions, bi_reactions).
    """
    uni_reactions = []
    bi_reactions = []
    idx = 0

    with open(path) as f:
        for line in f:
            template = line.strip()
            if not template or template.startswith('#'):
                continue
            rxn = Reaction(template, index=idx)
            if rxn.num_reactants == 1 and rxn.num_products == 1:
                uni_reactions.append(UniReaction(template, index=idx))
            elif rxn.num_reactants == 2 and rxn.num_products == 1:
                # The current molecule could be either reactant position.
                # We create two BiReaction objects (one per position).
                bi_reactions.append(BiReaction(template, is_mol_first=True, index=idx))
                bi_reactions.append(BiReaction(template, is_mol_first=False, index=idx))
            else:
                # Skip templates with unusual reactant/product counts
                pass
            idx += 1

    return uni_reactions, bi_reactions
