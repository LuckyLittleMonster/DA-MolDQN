
import numpy as np
from rdkit import Chem
from agent import MultiMolecules
from synnet.synth_env import SynthEnv
from synnet.utils.predict_utils import mol_fp
import hyp

class SynthMultiMolecules(MultiMolecules):
    """
    Adapter class to make SynNet environment compatible with MolDQN's MultiMolecules interface.
    This replaces the atom-based action space with synthesis-path actions.
    """
    def __init__(self, args, device, **kwargs):
        # Initialize parent
        super(SynthMultiMolecules, self).__init__(args, device, **kwargs)
        self.args = args
        
        # Initialize SynNet Environment
        # Note: SynthEnv loads models and data internally
        self.synth_env = SynthEnv(
            device=device.type if hasattr(device, 'type') else str(device).split(':')[0], # robust device extraction
            path_to_checkpoints=args.synnet_ckpt_dir,
            path_to_reaction_templates=args.synnet_rxn_templates,
            path_to_building_blocks=args.synnet_building_blocks,
            path_to_embeddings=args.synnet_embeddings
        )
        print("SynthMultiMolecules: SynNet Environment Initialized.")

    def _wrap_state(self, mol):
        """
        Ensure the molecule has the necessary attributes for SynNet (e.g., _syn_tree).
        If it's a raw RDKit mol (from init), we might need to wrap it.
        SynthEnv.get_valid_actions expects a mol with _syn_tree or handles initialization.
        Actually, SynthEnv logic checks if _syn_tree exists. If not, it creates a new one.
        So passing standard Chem.Mol is fine for the start.
        """
        return mol

    def initialize(self):
        """Resets the environment to initial molecules."""
        # Reset states to initial molecules
        # self.init_mols are Chem.Mol objects
        self.states = [self._wrap_state(m) for m in self.init_mols]
        
        # Reset any other tracking variables if needed
        # Parent initialize might do things? MultiMolecules doesn't have a complex initialize.
        # But we valid actions cache or other things in parent are reset implicitly by replacing self.states
        # Reset reward history
        # We need to know batch size. self.states is populated.
        self.reward_history = {'reward': [[] for _ in self.states]}

    def calc_valid_actions(self):
        """
        Calculate valid next synthesis steps for all current molecules (states).
        Returns:
            valid_actions_batch: List of List of Chem.Mol (the next states)
            fingerprints_batch: List of List of fingerprints (for the next states)
        """
        valid_actions_batch = []
        fingerprints_batch = []
        
        for state_mol in self.states:
            # 1. Get valid next states from SynNet
            # Returns valid_mols (List[Chem.Mol]) and valid_fingerprints (List)
            next_mols, _ = self.synth_env.get_valid_actions(state_mol)
            
            # 2. Generate fingerprints for these next states
            fps = []
            for m in next_mols:
                # Use SynNet fingerprint settings (Morgan, r=2, n=4096)
                # But ensure compatibility with main_hpc.py 'observation_type'
                from rdkit.Chem import AllChem
                fp_bit = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=4096)
                
                if self.args.observation_type == 'list':
                    # Expects list of ON bit indices
                    fps.append(list(fp_bit.GetOnBits()))
                elif self.args.observation_type == 'numpy':
                    # Expects dense numpy array
                    arr = np.zeros((4096,), dtype=np.int8)
                    from rdkit import DataStructs
                    DataStructs.ConvertToNumpyArray(fp_bit, arr)
                    fps.append(arr)
                else: 
                    # Fallback to list or handle 'vector'/'rdkit' if needed
                    # For now assume list default
                    fps.append(list(fp_bit.GetOnBits()))
                
            valid_actions_batch.append(next_mols)
            fingerprints_batch.append(fps)
            
        return valid_actions_batch, fingerprints_batch

    def step(self, actions, rewards):
        """
        Take a step in the environment.
        Args:
            actions: List of Chem.Mol (the chosen next state for each parallel environment)
            rewards: List of rewards (not used for state update, but for logging)
        """
        # Update current states to the chosen next states
        self.states = actions
        
        # We don't need to call C++ step or anything complex
        # because the 'action' IS the next state molecule object itself.
        # Append rewards to history
        # rewards is a dict: {'reward': [r1, r2, ...], 'bde': [...], ...}
        # We only really care about 'reward' for main_hpc priority, but typically we store all.
        # But for get_path usage, it accesses rewards['reward'][i][-1].
        # So we must structure it as {'reward': [[r_step1...], [r_step1...]]}
        for k, v in rewards.items():
            if k not in self.reward_history:
                self.reward_history[k] = [[] for _ in self.states]
            for i, r in enumerate(v):
                self.reward_history[k][i].append(r)

    def get_path(self):
        """
        Returns the synthesis path for recording.
        In MolDQN this records the sequence of molecules.
        SynNet molecules have _syn_tree, we could potentially log that.
        For now, let's return the molecules themselves as the 'path'.
        """
        # Placeholder: construct path from current state
        # In standard MolDQN, 'path' is maintained? 
        # MultiMolecules doesn't seem to maintain a full history explicitly in self.
        # main_hpc.py handles recording paths via detailed logging if args are set.
        # But wait, environment.get_path() is called in main_hpc.py.
        # Checking parent class... Molecule class has self._path?
        
        # For simplicity, returning just current states for now. 
        # Ideally we should traverse _syn_tree to get history.
        paths = []
        for s in self.states:
            if hasattr(s, '_syn_tree'):
                # Extract chemicals from tree
                chemicals = s._syn_tree.chemicals
                paths.append([c.smiles for c in chemicals]) # Return list of SMILES? 
            else:
                paths.append([Chem.MolToSmiles(s)])
                
        return paths, self.reward_history 
        # Actually parent get_path signature is: path, rewards
