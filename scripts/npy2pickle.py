import sys
import pickle
import glob
import os
import numpy as np
def main():
    if len(sys.argv) < 2:
        print("Usage: python npy2pickle.py <npy_files>")
        sys.exit(1)
        
    for npy_file in sys.argv[1:]:
        pickle_file = npy_file.replace('.npy', '.pickle')
        data = np.load(npy_file)
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f)
        os.remove(npy_file)
        print(f"Converted {npy_file} to {pickle_file}")
    
    print(f'Successfully converted {len(sys.argv[1:])} files')

if __name__ == "__main__":
    main()