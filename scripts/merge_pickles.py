import sys
import pickle
import glob
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python merge_pickles.py \"<pattern_with_star>.pickle\"")
        print("Example: python merge_pickles.py \"A_*.pickle\"")
        sys.exit(1)

    input_pattern = sys.argv[1]

    try:
        prefix, suffix = input_pattern.split('*')
    except ValueError:
        print("Error: Input pattern must contain exactly one '*'.")
        sys.exit(1)

    output_file = prefix + suffix
    file_list = glob.glob(input_pattern)

    if not file_list:
        print(f"No files found matching pattern: {input_pattern}")
        sys.exit(0)

    if output_file in file_list:
        print(f"Warning: Output file {output_file} is part of the input files. Removing it from processing.")
        file_list.remove(output_file)

    if not file_list:
        print("No source files left to process after removing output file.")
        sys.exit(0)
        
    merged_data = {}
    processed_files = []
    len_prefix = len(prefix)
    len_suffix = len(suffix)

    print(f"Merging files into {output_file}...")

    for filename in file_list:
        try:
            if "_.pickle" in filename:
                continue
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            key = filename[len_prefix:]
            if suffix:
                key = key[:-len(suffix)]
            
            merged_data[key] = data
            processed_files.append(filename)
            print(f"  + Added '{key}' from {filename}")

        except pickle.UnpicklingError:
            print(f"  ! Skipping {filename}: Not a valid pickle file.")
            sys.exit(1)
        except Exception as e:
            print(f"  ! Error processing {filename}: {e}")
            sys.exit(1)

    if not merged_data:
        print("No data was successfully merged.")
        sys.exit(0)

    try:
        with open(output_file, 'wb') as f:
            pickle.dump(merged_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\nSuccessfully merged {len(merged_data)} items into {output_file}.")
    except Exception as e:
        print(f"\nError writing output file {output_file}: {e}")
        sys.exit(1)

    print("Deleting source files...")
    deleted_count = 0
    for filename in processed_files:
        try:
            os.remove(filename)
            deleted_count += 1
        except Exception as e:
            print(f"  ! Error deleting {filename}: {e}")
    
    print(f"Deleted {deleted_count} source files.")
    print("Done.")

if __name__ == "__main__":
    main()
