import pickle
import gzip
import os
import sys

def convert_pickle_to_gz(pickle_file):
    """Convert a pickle file to a gzipped pickle file."""
    if not os.path.exists(pickle_file):
        print(f"Not found: {pickle_file}")
        return False
    
    # Generate output filename
    if pickle_file.endswith('.pickle'):
        gz_file = pickle_file[:-7] + '.pickle.gz'
    elif pickle_file.endswith('.pkl'):
        gz_file = pickle_file[:-4] + '.pkl.gz'
    else:
        gz_file = pickle_file + '.gz'
    
    # Skip if already a .gz file
    if pickle_file.endswith('.gz'):
        print(f"Skipping (already compressed): {pickle_file}")
        return False
    
    # Skip if output file already exists
    if os.path.exists(gz_file):
        print(f"Skipping (output exists): {gz_file}")
        return False
    
    try:
        # Read pickle file
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # Write gzipped pickle file
        with gzip.open(gz_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Get file sizes for comparison
        original_size = os.path.getsize(pickle_file)
        compressed_size = os.path.getsize(gz_file)
        ratio = (1 - compressed_size / original_size) * 100
        
        print(f"Converted: {pickle_file} -> {gz_file}")
        print(f"  Size: {original_size:,} bytes -> {compressed_size:,} bytes ({ratio:.1f}% reduction)")
        return True
    
    except Exception as e:
        print(f"Error converting {pickle_file}: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pickle_to_gz.py <pickle_file1> [pickle_file2] ...")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    success_count = 0
    
    for file_path in file_paths:
        if convert_pickle_to_gz(file_path):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(file_paths)} files converted successfully")

