
"""
Dry Run script to verify that the newly trained SynNet models can generate valid molecules.
"""
import logging
from rdkit import Chem
from synnet.synth_env import SynthEnv
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=== Starting Dry Run with New Models ===")
    
    # Initialize Environment with new checkpoints
    # Note: SynthEnv will look for checkpoints in this directory.
    # It expects: act.ckpt, rt1.ckpt, rxn.ckpt, rt2.ckpt (or directories with them)
    ckpt_dir = "synnet/checkpoints"
    
    print(f"Initializing SynthEnv from {ckpt_dir}...")
    try:
        env = SynthEnv(
            path_to_reaction_templates="Data/synnet/preprocessed/rxn_collection.json.gz",
            path_to_building_blocks="Data/synnet/preprocessed/zinc_filtered.csv.gz",
            path_to_embeddings="Data/synnet/preprocessed/building_blocks_emb.npy",
            path_to_checkpoints=ckpt_dir
        )
    except Exception as e:
        print(f"FAILED to initialize SynthEnv: {e}")
        sys.exit(1)
        
    print("SynthEnv initialized successfully.")
    
    # Generate a few molecules
    print("\n=== Generating 10 Molecules ===")
    
    # We will just simulate the "Environment" stepping.
    # In reality, the Agent drives this, but SynthEnv doesn't have a "generate_random_molecule" 
    # method publically exposed easily, but we can simulate the "valid action enumeration"
    # and pick the first action repeatedly until completion.
    
    # Or better: Just use the internal generate loop if available, or just test one step.
    # Let's test "get_valid_actions" like the Agent does.
    
    # Start with a random building block
    state = env.reset()
    print(f"Initial State: {Chem.MolToSmiles(state)}")
    
    valid_mols, _ = env.get_valid_actions(state)
    print(f"Generated {len(valid_mols)} valid next states from initial state.")
    
    if len(valid_mols) > 0:
        print(f"Sample Next State 0: {Chem.MolToSmiles(valid_mols[0])}")
        
        # Take a step
        state = valid_mols[0]
        # Try another step
        valid_mols_2, _ = env.get_valid_actions(state)
        print(f"Generated {len(valid_mols_2)} valid next states from step 1.")
        
        if len(valid_mols_2) > 0:
             print(f"Sample Next State 0 (Step 2): {Chem.MolToSmiles(valid_mols_2[0])}")

    print("\n=== Dry Run Complete ===")

if __name__ == "__main__":
    main()
