import argparse
import shutil
import os

TGLF_KEYS_FOR_REMOVAL = [
    'BT_EXP',
    'NN_MAX_ERROR',
    'VNS_SHEAR_1',
    'VNS_SHEAR_2',
    'RLNP_CUTOFF',
    'VTS_SHEAR_3',
    'DAMP_SIG',
    'VTS_SHEAR_2',
    'RHO_E',
    'VTS_SHEAR_1',
    'B_UNIT',
    'SAT_GEO0_OUT',
    'VNS_SHEAR_3',
    'WDIA_TRAPPED',
    'SHAPE_COS0',
    'SHAPE_COS1',
    'SHAPE_C0S2',
    'SHAPE_S_COS0',
    'SHAPE_S_COS1',
    'SHAPE_S_COS2',
    'KY'
]

TGLF_KEYS_TO_REPLACE = {
    'USE_MHD_RULE': 'T',
    'USE_BPAR': 'F',
    'WRITE_WAVEFUNCTION_FLAG': 1,
    'NKY': 24
}
import os
from typing import List

def refactor_tglf_file(filepath: str, prefixes_to_remove: List[str], to_replace: dict):
    """
    Reads TGLF file, removes any lines that start with a string found in
    the prefixes_to_remove list, replaces entries with keys in the to_replace_list, 
    and writes the modified content back to the original file path.

    Args:
        filepath (str): The path to the file to be processed.
        prefixes_to_remove (List[str]): A list of string prefixes. Any line
                                        starting with one of these prefixes
                                        (not case-sensitive) will be removed.
        to_replace (dict): A dictionary containing prefixes to be replaced as keys and values with the replacement
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at path: {filepath}")
        return

    filtered_lines = []

    try:
        # 1. Read all lines from the file
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 2. Filter the lines
        for line in lines:
            # Strip leading/trailing whitespace (including newline) for accurate prefix checking
            # but keep the original line to write back, preserving its newline character.
            stripped_line = line.lstrip().upper()

            # Check if the stripped line starts with any of the prefixes
            should_remove = any(stripped_line.startswith(prefix.upper()) for prefix in prefixes_to_remove)
            r_key= None
            r_val = None
            for prefix in to_replace.keys:
                should_replace = stripped_line.startswith(prefix.upper())
                if should_replace:
                    r_key = prefix
                    r_val = to_replace[prefix]
                    break

            if not should_remove:
                filtered_lines.append(line)
            
            if should_replace:
                filtered_lines.append(r_key + " = " + r_val)

        # 3. Write the filtered content back to the original file (overwriting it)
        with open(filepath, 'w') as f:
            f.writelines(filtered_lines)

        print(f"Successfully processed and updated file: {filepath}")

    except IOError as e:
        print(f"An error occurred while reading or writing the file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_dir")
    parser.add_argument("-d", "--destination_dir")

    args = parser.parse_args()
    source = args.source_dir
    dest = args.destination_dir
    try:
        shutil.copytree(source, dest)
        print(f"Directory '{source}' successfully copied to '{dest}'.")
        _, _, files = os.walk(dest)
        for file in files:
            refactor_tglf_file(file, TGLF_KEYS_FOR_REMOVAL, TGLF_KEYS_TO_REPLACE)
            print(f'Refactored TGLF file at {file}')
    except FileExistsError:
        print(f"Error: Destination directory '{dest}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")