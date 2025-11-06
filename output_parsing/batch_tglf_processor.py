#!/usr/bin/env python3
"""
Batch TGLF flux calculation processor.
Processes all batch-XXX directories and saves results to a single HDF5 file.
"""
import argparse
import os
import sys
import re
import numpy as np
import xarray as xr
import h5py
from pathlib import Path

# Import the functions from run_tglf_flux.py
from run_tglf_flux import (
    load_tglf_eigenvalue_spectrum,
    load_tglf_ql_spectrum,
    load_tglf_input_params,
    prepare_tglf_inputs,
    append_to_h5_individual_keys,
)

from qlgyro_and_tglf_flux_calculation import (
    get_sat_params,
    sum_ky_spectrum,
)


def find_batch_directories(root_dir):
    """
    Find all batch-XXX directories in root_dir.
    Returns sorted list of (batch_number, batch_path) tuples.
    """
    batch_pattern = re.compile(r'^batch-(\d+)$')
    batches = []
    
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            match = batch_pattern.match(item)
            if match:
                batch_num = int(match.group(1))
                batches.append((batch_num, item_path))
    
    # Sort by batch number
    batches.sort(key=lambda x: x[0])
    return batches


def find_tglf_files(batch_dir):
    """
    Find required TGLF files in batch directory.
    Returns dict with paths or None if files not found.
    """
    files = {
        'input': None,
        'eigenvalue': None,
        'ql_flux': None,
    }
    
    # Look for files with standard naming
    for filename in os.listdir(batch_dir):
        filepath = os.path.join(batch_dir, filename)
        if not os.path.isfile(filepath):
            continue
            
        # Match input file
        if filename == 'input.tglf' or filename == 'inputtglf' or filename == 'input_tglf':
            files['input'] = filepath
        # Match eigenvalue spectrum
        elif 'eigenvalue_spectrum' in filename:
            files['eigenvalue'] = filepath
        # Match QL flux spectrum
        elif 'QL_flux_spectrum' in filename:
            files['ql_flux'] = filepath
    
    # Check if all required files found
    if all(files.values()):
        return files
    else:
        missing = [k for k, v in files.items() if v is None]
        print(f"  ⚠️  Missing files: {missing}")
        return None


def process_single_batch(batch_num, batch_dir, sat_rule, alpha_zf):
    """
    Process a single batch directory.
    Returns tuple of (fluxes, sumf, ky, in0) or None if failed.
    """
    print(f"\n📂 Processing batch-{batch_num:03d}")
    
    # Find TGLF files
    files = find_tglf_files(batch_dir)
    if files is None:
        return None
    
    try:
        # Load TGLF output data
        print("  Loading TGLF outputs...")
        tglf_growth = load_tglf_eigenvalue_spectrum(files['eigenvalue'])
        tglf_ql = load_tglf_ql_spectrum(files['ql_flux'])
        
        nky = len(tglf_growth['ky'])
        print(f"    ✓ Loaded {nky} ky points")
        
        # Prepare inputs
        print("  Preparing input parameters...")
        in0 = prepare_tglf_inputs(batch_dir, sat_rule, alpha_zf)
        
        # Extract growth rates
        tglf_kys = tglf_growth['ky'].values
        tglf_growth_rates = tglf_growth['gamma'].values
        
        # Create growth rate array for all modes
        nmodes = tglf_ql['particle_flux'].shape[1]
        tglf_gammas = np.zeros((len(tglf_kys), nmodes))
        tglf_gammas[:, 0] = tglf_growth_rates
        
        # Calculate saturation parameters
        print("  Calculating saturation parameters...")
        gammas_for_sat = tglf_gammas.T
        kx0_e, satgeo1, satgeo2, R_unit, bt0, bgeo0, gradr0, _, _, _, _ = get_sat_params(
            sat_rule, tglf_kys, gammas_for_sat, **in0
        )
        
        # Update in0 with calculated geometry parameters
        in0['SAT_geo1_out'] = satgeo1
        in0['SAT_geo2_out'] = satgeo2
        in0['B_geo0_out'] = bgeo0
        in0['Bt0_out'] = bt0
        in0['grad_r0_out'] = gradr0
        
        # Extract QL weights
        particle_QL = tglf_ql['particle_flux'].values
        energy_QL = tglf_ql['energy_flux'].values
        toroidal_stress_QL = tglf_ql['toroidal_stress'].values
        parallel_stress_QL = tglf_ql['parallel_stress'].values
        exchange_QL = tglf_ql['exchange'].values
        
        # Run flux calculation
        print("  Running flux calculation...")
        tglf_sat = sum_ky_spectrum(
            sat_rule,
            tglf_kys,
            tglf_gammas,
            np.zeros_like(tglf_kys),
            R_unit,
            kx0_e,
            np.zeros((len(tglf_kys), nmodes)),
            particle_QL,
            energy_QL,
            toroidal_stress_QL,
            parallel_stress_QL,
            exchange_QL,
            **in0
        )
        
        # Sum over modes and fields to get total fluxes per species
        tglf_satG = np.sum(np.sum(tglf_sat['particle_flux_integral'], axis=2), axis=0)
        tglf_satQ = np.sum(np.sum(tglf_sat['energy_flux_integral'], axis=2), axis=0)
        tglf_satP = np.sum(np.sum(tglf_sat['toroidal_stresses_integral'], axis=2), axis=0)
        
        # Compute scalar fluxes: G_e, Q_e, Q_i, P_i
        G_elec = float(tglf_satG[0])
        Q_elec = float(tglf_satQ[0])
        Q_ions = float(np.sum(tglf_satQ[1:]))
        P_ions = float(np.sum(tglf_satP[1:]))
        
        fluxes = np.array([G_elec, Q_elec, Q_ions, P_ions], dtype=np.float32)
        
        # Sum flux spectrum over modes: (nky, nmodes, ns, nf, 5) -> (nky, ns, nf, 5)
        # sumf = np.sum(tglf_sat['sum_flux_spectrum'], axis=1).astype(np.float32)

        # swap ns and nf around (nky, nmodes, ns, nf, 5) -> (nky, nmodes, nf, ns, 5)
        sumf = np.transpose(tglf_sat['sum_flux_spectrum'].astype(np.float32), (0, 1, 3, 2, 4))
        
        ky_array = tglf_kys.astype(np.float32)
        
        print(f"  ✅ Fluxes: G_e={G_elec:.6f}, Q_e={Q_elec:.6f}, Q_i={Q_ions:.6f}, P_i={P_ions:.6f}")
        
        return fluxes, sumf, ky_array, in0
        
    except Exception as e:
        print(f"  ❌ Error processing batch: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_input_dict(in0):
    """
    Prepare input dictionary with required TGLF keys for HDF5 storage.
    """
    TGLF_KEYS = [
        "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1", "TAUS_2",
        "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1", "RLTS_2", "S_DELTA_LOC",
        "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3", "RLNS_3", "DZMAJDX_LOC", "DELTA_LOC",
        "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR", "RMAJ_LOC", "AS_2", "RLNS_2", "S_ZETA_LOC",
        "BETAE_log10", "XNUE_log10", "DEBYE_log10"
    ]
    
    input_dict = {k: in0.get(k, np.nan) for k in TGLF_KEYS}
    
    # Compute log10 values if base values exist
    for key in ["BETAE", "XNUE", "DEBYE"]:
        if key in in0:
            input_dict[f"{key}_log10"] = np.log10(in0[key])
    
    return input_dict


def process_all_batches(root_dir, output_h5, sat_rule=2, alpha_zf=1.0, 
                        start_batch=None, end_batch=None):
    """
    Process all batch directories and save to a single HDF5 file.
    
    Parameters:
    -----------
    root_dir : str
        Root directory containing batch-XXX subdirectories
    output_h5 : str
        Path to output HDF5 file
    sat_rule : int
        Saturation rule (1, 2, or 3)
    alpha_zf : float
        Zonal flow coefficient
    start_batch : int, optional
        Start processing from this batch number
    end_batch : int, optional
        Stop processing at this batch number (inclusive)
    """
    # Find all batch directories
    batches = find_batch_directories(root_dir)
    
    if not batches:
        print(f"❌ No batch directories found in {root_dir}")
        return
    
    print(f"📊 Found {len(batches)} batch directories")
    
    # Filter by start/end if specified
    if start_batch is not None:
        batches = [(n, p) for n, p in batches if n >= start_batch]
    if end_batch is not None:
        batches = [(n, p) for n, p in batches if n <= end_batch]
    
    print(f"📝 Processing {len(batches)} batches (SAT{sat_rule}, α_ZF={alpha_zf})")
    
    # Remove existing HDF5 file if it exists
    if os.path.exists(output_h5):
        print(f"⚠️  Removing existing file: {output_h5}")
        os.remove(output_h5)
    
    # Process each batch
    success_count = 0
    fail_count = 0
    
    for batch_num, batch_path in batches:
        result = process_single_batch(batch_num, batch_path, sat_rule, alpha_zf)
        
        if result is not None:
            fluxes, sumf, ky, in0 = result
            
            # Prepare input dictionary
            input_dict = prepare_input_dict(in0)
            
            # Save to HDF5
            try:
                append_to_h5_individual_keys(
                    h5_path=output_h5,
                    input_dict=input_dict,
                    fluxes=fluxes,
                    sumf=sumf,
                    ky=ky,
                    meta={
                        'batch_num': batch_num,
                        'sat_rule': sat_rule,
                        'alpha_zf': alpha_zf,
                        'nky': len(ky),
                        'nspecies': sumf.shape[1],
                    }
                )
                success_count += 1
            except Exception as e:
                print(f"  ❌ Error saving to HDF5: {e}")
                fail_count += 1
        else:
            fail_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed: {success_count}/{len(batches)} batches")
    if fail_count > 0:
        print(f"❌ Failed: {fail_count}/{len(batches)} batches")
    print(f"💾 Results saved to: {output_h5}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process TGLF flux calculations and save to single HDF5 file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process all batches in directory
  python batch_tglf_flux_processor.py reformatted_tglf_inputs -o results.h5
  
  # Use SAT3 instead of SAT2
  python batch_tglf_flux_processor.py reformatted_tglf_inputs -o results.h5 --sat_rule 3
  
  # Process only batches 0-10
  python batch_tglf_flux_processor.py reformatted_tglf_inputs -o results.h5 --start 0 --end 10
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Root directory containing batch-XXX subdirectories"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="tglf_batch_results.h5",
        help="Output HDF5 file path (default: tglf_batch_results.h5)"
    )
    
    parser.add_argument(
        "--sat_rule",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Saturation rule (default: 2)"
    )
    
    parser.add_argument(
        "--alpha_zf",
        type=float,
        default=1.0,
        help="Zonal flow coupling coefficient (default: 1.0)"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start from this batch number (optional)"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End at this batch number (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"❌ Error: Directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Process all batches
    process_all_batches(
        root_dir=args.input_dir,
        output_h5=args.output,
        sat_rule=args.sat_rule,
        alpha_zf=args.alpha_zf,
        start_batch=args.start,
        end_batch=args.end
    )


if __name__ == "__main__":
    main()