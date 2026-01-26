
import os
import sys
import numpy as np
import torch
import pytorch_lightning as pl
from rdkit import Chem
from typing import List, Tuple
from sklearn.neighbors import BallTree

# Fix import path to allow running script directly from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
# Ensure the root of DA-MolDQN is in sys.path when running this
# We use absolute imports assuming 'synnet' package is importable
from synnet.utils.data_utils import SyntheticTree, ReactionSet, Reaction, NodeChemical
from synnet.utils.predict_utils import mol_fp, get_action_mask, get_reaction_mask, can_react, set_embedding, nn_search
from synnet.models.common import load_mlp_from_ckpt
from synnet.MolEmbedder import MolEmbedder
from synnet.encoding.distances import cosine_distance
from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.encoding.utils import one_hot_encoder

class SynthEnv:
    """
    Wrapper environment for SynNet to be used by MolDQN.
    Handles loading models, maintaining chemical databases, and enumerating valid actions.
    """
    def __init__(self, 
                 device: str = "cpu",
                 data_dir: str = None,
                 ckpt_dir: str = None,
                 rxn_template_file: str = "rxns_hb.json.gz",
                 bb_file: str = "building_blocks_matched.csv.gz",
                 embeddings_knn_file: str = "building_blocks_emb.npy",
                 path_to_reaction_templates: str = None, # Alias usually used
                 path_to_building_blocks: str = None,
                 path_to_embeddings: str = None,
                 path_to_checkpoints: str = None
                 ):
        
        self.device = device
        
        # Determine Data Directory
        # Determine Data Directory
        # Default to the centralized data directory
        default_data_dir = os.path.abspath("Data/synnet")
        self.data_dir = data_dir if data_dir else default_data_dir
        
        # Determine Checkpoint Directory
        # Default to synnet/checkpoints if not provided
        default_ckpt_dir = os.path.abspath("synnet/checkpoints")
        self.ckpt_dir = path_to_checkpoints if path_to_checkpoints else (ckpt_dir if ckpt_dir else default_ckpt_dir)
        
        # Determine Asset Directory (assumed relative to data_dir usually, or explicit)
        self.asset_dir = os.path.join(self.data_dir, "assets") # Default asset dir
        
        # Allow explicit full paths to override
        self.rxn_template_path = path_to_reaction_templates if path_to_reaction_templates else os.path.join(self.ckpt_dir, rxn_template_file)
        self.bb_path = path_to_building_blocks if path_to_building_blocks else os.path.join(self.asset_dir, bb_file)
        self.emb_path = path_to_embeddings if path_to_embeddings else os.path.join(self.ckpt_dir, embeddings_knn_file)

        print(f"Initializing SynthEnv...")
        print(f"  Ckpt Dir: {self.ckpt_dir}")
        print(f"  Rxn Template: {self.rxn_template_path}")


        # 1. Load Data
        # 1. Load Data
        self._load_data()
        
        # 2. Load Models
        self._load_models()
        
        print("SynthEnv initialized successfully.")

    def reset(self):
        """Resets the environment to an empty state."""
        # Create an empty molecule to represent the start of synthesis
        mol = Chem.MolFromSmiles("")
        # Attach a new, empty Synthetic Tree
        mol._syn_tree = SyntheticTree()
        return mol

    def _load_data(self):
        """Loads BBs, Templates, and Embeddings."""
        # A. Load Reaction Templates
        if self.rxn_template_path.endswith('.txt'):
             from synnet.data_generation.preprocessing import ReactionTemplateFileHandler
             from synnet.utils.data_utils import Reaction
             rxn_templates = ReactionTemplateFileHandler().load(self.rxn_template_path)
             self.rxns = [Reaction(template=t) for t in rxn_templates]
        else:
             self.rxns = ReactionSet().load(self.rxn_template_path).rxns
             
        self.rxn_one_hot_size = len(self.rxns) 
        print(f"  Loaded {len(self.rxns)} reaction templates (One-Hot Size: {self.rxn_one_hot_size})")

        # B. Load Building Blocks
        if not os.path.exists(self.bb_path):
             raise FileNotFoundError(f"Building Block file {self.bb_path} not found.")
        self.building_blocks = BuildingBlockFileHandler().load(self.bb_path)
        self.bb_dict = {bb: i for i, bb in enumerate(self.building_blocks)}

        # C. Load Embeddings and Build Tree
        if not os.path.exists(self.emb_path):
             raise FileNotFoundError(f"Embedding file {self.emb_path} not found.")
        self.bb_emb = np.load(self.emb_path)
        
        print("  Building BallTree for k-NN search...")
        self.kdtree = BallTree(self.bb_emb, metric=cosine_distance)

    def _load_models(self):
        """Loads the 4 MLP models."""
        self.models = {}
        for name in ["act", "rt1", "rxn", "rt2"]:
            ckpt_path = os.path.join(self.ckpt_dir, f"{name}.ckpt")
            if not os.path.exists(ckpt_path):
                 raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
            model = load_mlp_from_ckpt(ckpt_path)
            model.eval()
            model.to(self.device)
            self.models[name] = model

        self.act_net = self.models["act"]
        self.rt1_net = self.models["rt1"]
        self.rxn_net = self.models["rxn"]
        self.rt2_net = self.models["rt2"]

    def get_valid_actions(self, 
                          mol: Chem.Mol, 
                          top_k_actions: int = 2,
                          top_k_rxns: int = 2,
                          top_k_reactants: int = 2) -> Tuple[List[Chem.Mol], List[str]]:
        """
        Enumerates valid next steps (Action, Rxn, Reactant) from the current state.
        Args:
            mol: Current RDKit molecule (state). MUST have ._syn_tree attribute!
        Returns:
            valid_mols: List of next-state RDKit molecules (with ._syn_tree attached).
            valid_fingerprints: List of corresponding fingerprints (optional, logic might move to C++).
        """
        if not hasattr(mol, "_syn_tree"):
             # If starting from scratch/empty, create a new tree
             tree = SyntheticTree()
             if mol.GetNumAtoms() > 0:
                 smi = Chem.MolToSmiles(mol)
                 node = NodeChemical(
                     smiles=smi,
                     is_leaf=True,
                     is_root=True,
                     depth=0,
                     index=0
                 )
                 tree.chemicals.append(node)
                 tree.root = node
        else:
             tree = mol._syn_tree

        state = tree.get_state()
        
        # 1. Encode State
        # Target Embedding: In RL, we might use a placeholder or the Agent's goal.
        # SynNet requires a 'z_target'. If we don't have one (unconditional), we can pass zeros or current state.
        # For 'Generation', we often assume we decode *towards* a target. 
        # But here the Agent *is* the navigator. Providing a dummy target might bias it?
        # Let's use numpy zeros of size 4096 for z_target (fingerprint size)
        z_target = np.zeros((1, 4096)) 
        
        z_state = set_embedding(z_target, state, nbits=4096, _mol_embedding=mol_fp)
        z_state_torch = torch.Tensor(z_state).to(self.device)

        valid_next_mols=[]
        
        # 2. Predict Probabilities for ACTION
        action_proba = self.act_net(z_state_torch).detach().cpu().numpy().squeeze() + 1e-10
        action_mask = get_action_mask(state, self.rxns)
        masked_action_proba = action_proba * action_mask
        
        # Get Top-K Actions
        # Filter actions with 0 probability
        valid_action_indices = np.argwhere(masked_action_proba > 0).flatten()
        # Sort by prob
        sorted_action_indices = valid_action_indices[np.argsort(masked_action_proba[valid_action_indices])[::-1]]
        top_actions = sorted_action_indices[:top_k_actions]
        
        for act in top_actions:
            if act == 3: # End
                # Treat End as returning the current molecule (idempotent/terminal)
                # We clone it to be safe
                import copy
                new_mol = copy.deepcopy(mol) # Deep copy to preserve tree
                # We might want to mark it as done, but for now just return it as a valid state.
                valid_next_mols.append(new_mol)
                continue 

            # Prepare for Reactant 1 prediction
            # act=0: Add, act=1: Expand, act=2: Merge
            
            mol1 = None
            if act == 0: # Add
                z_mol1_emb = self.rt1_net(z_state_torch).detach().cpu().numpy()
                # KNN Search
                dist, idxs = self.kdtree.query(z_mol1_emb, k=top_k_reactants)
                possible_mol1s = [self.building_blocks[i] for i in idxs[0]]
                
            elif act == 1 or act == 2: # Expand or Merge
                # Use most recent molecule
                possible_mol1s = [state[-1]] # List of 1
            
            for m1 in possible_mol1s:
                z_m1 = mol_fp(m1)
                z_m1 = np.atleast_2d(z_m1)
                
                # Predict Reaction
                z_rxn_input = np.concatenate([z_state, z_m1], axis=1)
                rxn_proba = self.rxn_net(torch.Tensor(z_rxn_input).to(self.device)).detach().cpu().numpy().squeeze() + 1e-10
                
                # Mask Reactions
                if act == 0 or act == 1:
                     mask, avail = get_reaction_mask(m1, self.rxns)
                     if mask is None: continue
                else: # Merge
                     count, mask = can_react(state, self.rxns)
                     if count == 0: continue
                     avail = [[] for _ in self.rxns] # Not used for Merge
                
                masked_rxn_proba = rxn_proba * np.array(mask)
                valid_rxn_indices = np.argwhere(masked_rxn_proba > 0).flatten()
                top_rxns = valid_rxn_indices[np.argsort(masked_rxn_proba[valid_rxn_indices])[::-1]][:top_k_rxns]
                
                for r_idx in top_rxns:
                    rxn_obj = self.rxns[r_idx]
                    
                    # Predict Reactant 2?
                    possible_mol2s = [None]
                    if rxn_obj.num_reactant == 2:
                        if act == 2: # Merge
                             # Merge uses existing molecules in state
                             # state has at least 2 mols. mol1 is state[-1]. 
                             # mol2 must be the OTHER root. (Assumes 2 roots max for Merge logic?)
                             # SynNet state is list of roots.
                             # If Merge, pop 2.
                             # predict_utils logic: temp = set(state) - set([mol1]) -> pop
                             remaining = list(set(state) - set([m1]))
                             if not remaining: continue 
                             possible_mol2s = [remaining[0]]
                        else: # Add/Expand with 2nd reactant
                             # Predict Reactant 2
                             # Inputs: State + Mol1 + Rxn (One-Hot)
                             # Use self.rxn_one_hot_size instead of 91
                             x_rxn = one_hot_encoder(r_idx, self.rxn_one_hot_size) 
                             x_rct2 = np.concatenate([z_state, z_m1, x_rxn], axis=1)
                             z_rct2 = self.rt2_net(torch.Tensor(x_rct2).to(self.device)).detach().cpu().numpy()
                             
                             # Masked KNN
                             # available_reactants is list of list of strings
                             # We need to intersect available reactants for THIS reaction with our BB list
                             # This is slow! 
                             # predict_utils uses `available_list` which comes from `get_reaction_mask`
                             # `get_reaction_mask` returns `available_list` where `available_list[i]` is list of allowed reactants.
                             
                             # Optimization: The `available_list` returned by `get_reaction_mask` ALREADY contains the valid BBs.
                             # But `available_list` is list of strings. We need their indices to use `bb_emb`?
                             # Or we just kNN on the embedding and check if result is in `Available`?
                             # predict_utils: 
                             #   available = available_list[rxn_id]
                             #   available_indices = [bb_dict[s] for s in available]
                             #   Build temporary BallTree from ONLY available embeddings.
                             
                             allowed_bbs = avail[r_idx]
                             if not allowed_bbs: continue
                             
                             allowed_indices = [self.bb_dict[s] for s in allowed_bbs if s in self.bb_dict]
                             if not allowed_indices: continue
                             
                             temp_emb = self.bb_emb[allowed_indices]
                             # Build tiny tree? Or brute force?
                             # For top-K, brute force distance might be faster if N is small
                             # But let's stick to simple logic
                             local_tree = BallTree(temp_emb, metric=cosine_distance)
                             d, ind = local_tree.query(z_rct2, k=min(len(allowed_indices), top_k_reactants))
                             
                             possible_mol2s = [self.building_blocks[allowed_indices[idx]] for idx in ind[0]]

                    for m2 in possible_mol2s:
                        # Run Reaction
                        prod = rxn_obj.run_reaction((m1, m2))
                        if prod:
                             # Create New Tree State (Clone and Update)
                             import copy
                             new_tree = copy.deepcopy(tree)
                             new_tree.update(act, r_idx, m1, m2, prod)
                             
                             # New Mol
                             new_mol = Chem.MolFromSmiles(prod)
                             if new_mol:
                                 new_mol._syn_tree = new_tree
                                 valid_next_mols.append(new_mol)

        if not valid_next_mols:
             # Fallback: If no actions generated (dead end), return current mol
             # This prevents main_hpc crash and allows max_steps to terminate
             import copy
             valid_next_mols.append(copy.deepcopy(mol))

        return valid_next_mols, []

if __name__ == "__main__":
    # Smoke Test
    try:
        env = SynthEnv()
        # Create empty mol 
        m = Chem.MolFromSmiles("") 
        # Attach empty tree
        m._syn_tree = SyntheticTree()
        
        print("Testing Action Enumeration...")
        actions, _ = env.get_valid_actions(m)
        print(f"Generated {len(actions)} valid next states.")
        if actions:
            print(f"Sample state 0: {Chem.MolToSmiles(actions[0])}")
            print(f"Tree Depth: {actions[0]._syn_tree.depth}")
        print("Integration Test Passed.")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
