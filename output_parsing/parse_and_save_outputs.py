import os
import h5py
import numpy as np
import argparse
import re

# Matches subdirectories like: input, input-0, input-0007, input_12
INPUT_DIR_PAT = re.compile(r"^input(?:[-_]?)(\d+)?$")

def _sort_key_for_input_dir(name: str):
    """
    Sort by numeric suffix if present (0,1,2,...), else lexicographic after numerics.
    """
    m = INPUT_DIR_PAT.match(name)
    if m and m.group(1) is not None:
        try:
            return (0, int(m.group(1)), name)  # numeric dirs first
        except ValueError:
            pass
    return (1, name, name)  # non-numeric fall back

def _parse_ky_from_file(path: str) -> float | None:
    """
    Parse 'KY = <float>' (or 'KY=<float>') from a CGYRO input file.
    - Ignores case and surrounding spaces.
    - Tolerates Fortran 'D' exponents by converting to 'E'.
    - Ignores inline comments after the value.
    """
    try:
        with open(path, "r") as f:
            for line in f:
                if "KY" not in line:
                    continue
                parts = line.strip().split("=")
                if len(parts) != 2:
                    continue
                key = parts[0].strip().upper()
                if key != "KY":
                    continue
                # Strip inline comments (# or !), keep first token as value
                val_part = parts[1].split("#", 1)[0].split("!", 1)[0].strip()
                # Handle Fortran D-exponents (e.g., 1.0D+03)
                val_part = val_part.replace("D", "E").replace("d", "e")
                # Value may have trailing tokens; take the first whitespace-separated token
                val_tok = val_part.split()[0]
                return float(val_tok)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None

def read_ky_list_from_cgyro_inputs(cgyro_dir: str) -> np.ndarray:
    """
    Walks `cgyro_dir` for subdirectories named 'input*' (e.g., input-000, input_001),
    expects each to contain an 'input.cgyro' file, and extracts one KY per subdir.

    Returns
    -------
    np.ndarray
        shape (N,), dtype float32, ordered by numeric suffix.
        Missing/unparsable KY becomes np.nan (with a warning).
    """
    # 1) Collect candidate subdirectories directly under cgyro_dir
    try:
        entries = os.listdir(cgyro_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {cgyro_dir}")

    input_dirs = [
        d for d in entries
        if os.path.isdir(os.path.join(cgyro_dir, d)) and INPUT_DIR_PAT.match(d)
    ]
    if not input_dirs:
        raise FileNotFoundError(f"No input-* subdirectories found under {cgyro_dir}")

    # 2) Sort them stably: numeric suffix first, ascending; then lexicographic
    input_dirs_sorted = sorted(input_dirs, key=_sort_key_for_input_dir)

    # 3) For each subdir, parse KY from its input.cgyro
    ky_list = []
    missing = []
    for d in input_dirs_sorted:
        subdir = os.path.join(cgyro_dir, d)
        # Prefer exact 'input.cgyro'
        primary = os.path.join(subdir, "input.cgyro")
        ky = _parse_ky_from_file(primary)
        if ky is None:
            # Fallback: search for any file that starts with 'input' in that subdir
            candidates = [
                os.path.join(subdir, f)
                for f in os.listdir(subdir)
                if os.path.isfile(os.path.join(subdir, f)) and f.startswith("input")
            ]
            ky = None
            for c in candidates:
                ky = _parse_ky_from_file(c)
                if ky is not None:
                    break

        if ky is None:
            missing.append(d)
            ky_list.append(np.nan)
        else:
            ky_list.append(float(ky))

    if missing:
        print(f"⚠️ Missing KY in {len(missing)} subdirs (set to NaN): {missing[:8]}{'...' if len(missing) > 8 else ''}")

    return np.asarray(ky_list, dtype=np.float32)


# === List of keys to extract from TGLF input ===
TGLF_KEYS = [
    "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1", "TAUS_2",
    "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1", "RLTS_2", "S_DELTA_LOC",
    "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3", "RLNS_3", "DZMAJDX_LOC", "DELTA_LOC",
    "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR", "RMAJ_LOC", "AS_2", "RLNS_2", "S_ZETA_LOC",
    "BETAE_log10", "XNUE_log10", "DEBYE_log10"
]

def parse_tglf_file(filepath):
    vals = {}
    with open(filepath, 'r') as f:
        for line in f:
            if '=' not in line or line.strip().startswith("#"):
                continue
            key, val = map(str.strip, line.strip().split("="))
            if val.upper() in ("T", ".TRUE."):
                val = True
            elif val.upper() in ("F", ".FALSE."):
                val = False
            else:
                try:
                    val = float(val)
                except ValueError:
                    continue
            vals[key] = val

    # Derived log10 values
    for key in ["BETAE", "XNUE", "DEBYE"]:
        if key in vals:
            vals[f"{key}_log10"] = np.log10(vals[key])

    # Extract only the required keys
    inputs = {k: vals.get(k, np.nan) for k in TGLF_KEYS}
    sat_rule = int(vals.get("SAT_RULE", 2))
    return inputs, sat_rule


def append_to_h5_individual_keys(h5_path, input_dict, fluxes, sumf, ky, meta=None):
    import h5py
    import numpy as np

    fluxes = np.atleast_1d(fluxes).astype(np.float32)         # shape: (4,)
    sumf = np.asarray(sumf, dtype=np.float32)                 # shape: (total_count, ns, nf, 3)
    ky = np.asarray(ky, dtype=np.float32)                     # shape: (total_count,)

    with h5py.File(h5_path, "a") as f:
        print("📥 Appending:")
        print("  fluxes:", fluxes.shape)
        print("  sumf:", sumf.shape)
        print("  ky:", ky.shape)

        # === Save scalar input keys ===
        for name, data in input_dict.items():
            data = np.atleast_1d(data).astype(np.float32)
            if name not in f:
                f.create_dataset(name, data=data, maxshape=(None,), chunks=True)
            else:
                f[name].resize(f[name].shape[0] + 1, axis=0)
                f[name][-1] = data

        # === Save flux vector (G_e, Q_e, Q_i, P_i)
        flux_names = ["OUT_G_e", "OUT_Q_e", "OUT_Q_i", "OUT_P_i"]
        for i, name in enumerate(flux_names):
            val = np.float32(fluxes[i])
            if name not in f:
                f.create_dataset(name, data=[val], maxshape=(None,), chunks=True)
            else:
                f[name].resize((f[name].shape[0] + 1), axis=0)
                f[name][-1] = val

        # === Save sumf matrix: (1, total_count, ns, nf, 3)
        if "sumf" not in f:
            f.create_dataset("sumf", data=sumf[None, ...], maxshape=(None,) + sumf.shape, chunks=True)
        else:
            f["sumf"].resize((f["sumf"].shape[0] + 1), axis=0)
            f["sumf"][-1] = sumf

        # === Save ky array: (1, total_count)
        if "ky" not in f:
            f.create_dataset("ky", data=ky[None, :], maxshape=(None, ky.shape[0]), chunks=True)
        else:
            f["ky"].resize((f["ky"].shape[0] + 1), axis=0)
            f["ky"][-1] = ky

        # === Save meta info ===
        if meta:
            meta_grp = f.require_group("meta")
            for key, value in meta.items():
                value = np.asarray(value)
                if key not in meta_grp:
                    meta_grp.create_dataset(key, data=value[None, ...] if value.ndim > 0 else value[None],
                                            maxshape=(None,) + value.shape if value.ndim > 0 else (None,),
                                            chunks=True)
                else:
                    meta_grp[key].resize(meta_grp[key].shape[0] + 1, axis=0)
                    meta_grp[key][-1] = value




def process_and_save_flux_data(top_dir, h5_out_path):
    from output_parsing import parse_outputs
    import numpy as np
    import os

    for batch_name in sorted(os.listdir(top_dir)):
        batch_path = os.path.join(top_dir, batch_name)
        if not os.path.isdir(batch_path):
            continue

        cgyro_dir = os.path.join(batch_path, "cgyro")
        tglf_dir = os.path.join(batch_path, "tglf")

        # Find first TGLF input file
        tglf_input_file = None
        for root, dirs, files in os.walk(tglf_dir):
            for fname in files:
                if fname.endswith(".tglf"):
                    tglf_input_file = os.path.join(root, fname)
                    break
            if tglf_input_file:
                break

        if not tglf_input_file:
            print(f"❌ No TGLF input found in {batch_name}")
            continue

        # Parse TGLF (should return: dict, sat_rule)
        input_dict, sat_rule = parse_tglf_file(tglf_input_file)

        try:
            results, kys_valid, failed_indices, total_count = parse_outputs.run_qlgyro_flux_from_dir(
                root_dir=cgyro_dir,
                tglf_input_path=tglf_input_file,
                sat_rules=(2, 3),
                alpha_zf_in=1.0,
            )
        except Exception as e:
            print(f"⚠️ Failed {batch_name}: {e}")
            continue

        if sat_rule not in results:
            print(f"⚠️ Missing SAT rule {sat_rule} in results for {batch_name}")
            continue

        # Extract fluxes and sumf for SAT rule
        fluxes, sumf = results[sat_rule]
        fluxes = np.asarray(fluxes)  # (4,)
        sumf = np.asarray(sumf)      # (n_valid, ns, nf, 3)
        kys_valid = np.asarray(kys_valid)

        # === Construct valid -> full padded arrays ===
        valid_indices = [i for i in range(total_count) if i not in failed_indices]
        shape = sumf.shape[1:]  # (ns, nf, 3)

        full_sumf = np.zeros((total_count,) + shape, dtype=np.float32)
        failed_mask = np.zeros((total_count,), dtype=bool)

        for new_idx, original_idx in enumerate(valid_indices):
            full_sumf[original_idx] = sumf[new_idx]

        for i in failed_indices:
            failed_mask[i] = True

        # === Authoritative KY from CGYRO input files ===
        try:
            ky_all = read_ky_list_from_cgyro_inputs(cgyro_dir)  # shape: (total_count,)
            if ky_all.shape[0] != total_count:
                print(f"⚠️ KY count ({ky_all.shape[0]}) != total_count ({total_count}); "
                      f"will align to min length and pad NaNs.")
                # Align lengths safely
                full_kys = np.full((total_count,), np.nan, dtype=np.float32)
                n = min(total_count, ky_all.shape[0])
                full_kys[:n] = ky_all[:n]
            else:
                full_kys = ky_all.astype(np.float32)
        except Exception as e:
            print(f"⚠️ Falling back to kys_valid due to KY parse error: {e}")
            # Previous behavior: place kys_valid in valid slots only, rest zeros/NaN
            full_kys = np.full((total_count,), np.nan, dtype=np.float32)
            for new_idx, original_idx in enumerate(valid_indices):
                full_kys[original_idx] = kys_valid[new_idx].astype(np.float32)
        
        # === Save everything to HDF5 ===
        append_to_h5_individual_keys(
            h5_path=h5_out_path,
            input_dict=input_dict,
            fluxes=fluxes.astype(np.float32),
            sumf=full_sumf.astype(np.float32),
            ky=full_kys.astype(np.float32),
            meta=dict(
                total_count=total_count,
                failed_mask=failed_mask.astype(np.uint8)
            )
        )


# === Example Usage ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory")
    parser.add_argument("-o", "--output_path")

    args = parser.parse_args()
    dir = args.directory
    out = args.output_path
    process_and_save_flux_data(dir, out)