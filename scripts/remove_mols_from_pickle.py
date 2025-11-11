import pickle
import os
import sys
from rdkit.Chem.rdchem import Mol
from rdkit import Chem

def is_mol_string(s):
    """Check if a string can be converted to an rdkit Mol object."""
    if not isinstance(s, str):
        return False
    
    # Try different formats: SMILES, InChI, MolBlock, SMARTS
    converters = [
        Chem.MolFromSmiles,
        Chem.MolFromInchi,
        Chem.MolFromMolBlock,
        Chem.MolFromSmarts,
    ]
    
    for converter in converters:
        try:
            mol = converter(s)
            if mol is not None:
                return True
        except:
            continue
    
    return False

def sanitize_structure(data):
    if isinstance(data, dict):
        new_data = data.__class__()
        for key, value in data.items():
            new_data[key] = sanitize_structure(value)
        return new_data
    
    elif isinstance(data, list):
        new_list = data.__class__()
        for item in data:
            new_list.append(sanitize_structure(item))
        return new_list
    
    elif isinstance(data, tuple):
        new_tuple_list = []
        for item in data:
            new_tuple_list.append(sanitize_structure(item))
        return data.__class__(new_tuple_list)
    
    elif isinstance(data, Mol):
        return None
    
    elif isinstance(data, str):
        if is_mol_string(data):
            return None
        else:
            return data
    
    else:
        return data

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python remove_mols_from_pickle.py <pickle_file>")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Not found: {file_path}")
            continue
        with open(file_path, 'rb') as f:
            original_data = pickle.load(f)
        sanitized_data = sanitize_structure(original_data)
        with open(file_path, 'wb') as f:
            pickle.dump(sanitized_data, f)
        print(f"Removed mols from {file_path}")