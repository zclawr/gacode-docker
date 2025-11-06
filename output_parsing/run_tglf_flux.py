#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import xarray as xr
import h5py

# Import only the functions we need
from qlgyro_and_tglf_flux_calculation import (
    get_sat_params,
    sum_ky_spectrum,
)


def load_tglf_eigenvalue_spectrum(filepath):
    """
    Load TGLF eigenvalue spectrum from text file.
    Format: pairs of (gamma, freq) for each mode at each ky
    Returns xarray Dataset with ky, gamma, and freq arrays.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines
    data_lines = [line.strip() for line in lines[2:] if line.strip() and not line.strip().startswith('#')]
    
    # Parse the data - each line contains alternating gamma,freq pairs
    ky_list = []
    gamma_list = []
    freq_list = []
    
    for line in data_lines:
        values = [float(x) for x in line.split()]
        if len(values) == 0:
            continue
        
        # Values come in pairs: gamma1, freq1, gamma2, freq2, ...
        # We want the first (most unstable) mode
        if len(values) >= 2:
            gamma_list.append(values[0])
            freq_list.append(values[1])
    
    # Create ky grid (TGLF default: 21 points from 0.1 to 10.0 logarithmically)
    nky = len(gamma_list)
    ky_list = np.logspace(np.log10(0.1), np.log10(10.0), nky)
    
    return xr.Dataset({
        'ky': (['ky'], ky_list),
        'gamma': (['ky'], gamma_list),
        'freq': (['ky'], freq_list),
    })


def load_tglf_ql_spectrum(filepath):
    """
    Load TGLF QL flux spectrum from text file.
    
    Format from header:
    QL_flux_spectrum_out(type,nspecies,field,ky,mode)
    index limits: type,ns,field,nky,nmodes
    5  3  2  33  5
    
    The file is organized as:
    - For each species (1 to nspecies)
      - For each field (1 to nfield)
        - For each mode (1 to nmodes)
          - nky rows with 5 values each (particle, energy, toroidal_stress, parallel_stress, exchange)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header to get dimensions
    for i, line in enumerate(lines):
        if 'index limits' in line:
            dims_line = lines[i + 1].strip()
            ntypes, nspecies, nfield, nky, nmodes = [int(x) for x in dims_line.split()]
            break
    
    # Initialize arrays
    # Shape: [ky, mode, species, field, type]
    ql_data = np.zeros((nky, nmodes, nspecies, nfield, ntypes))
    
    # Parse data
    current_line = 0
    for line in lines:
        if line.strip().startswith('species ='):
            parts = line.split()
            species_idx = int(parts[2]) - 1  # Convert to 0-indexed
            field_idx = int(parts[5]) - 1
            current_line = lines.index(line) + 1
            
        elif line.strip().startswith('mode ='):
            mode_idx = int(line.split()[2]) - 1
            current_line = lines.index(line) + 1
            
            # Read nky rows of data
            data_count = 0
            for j in range(current_line, len(lines)):
                data_line = lines[j].strip()
                if not data_line or data_line.startswith('species') or data_line.startswith('mode'):
                    break
                
                values = [float(x) for x in data_line.split()]
                if len(values) == ntypes and data_count < nky:
                    ql_data[data_count, mode_idx, species_idx, field_idx, :] = values
                    data_count += 1
    
    # Create ky grid
    ky_values = np.logspace(np.log10(0.1), np.log10(10.0), nky)
    
    return xr.Dataset({
        'particle_flux': (['ky', 'mode', 'species', 'field'], ql_data[:, :, :, :, 0]),
        'energy_flux': (['ky', 'mode', 'species', 'field'], ql_data[:, :, :, :, 1]),
        'toroidal_stress': (['ky', 'mode', 'species', 'field'], ql_data[:, :, :, :, 2]),
        'parallel_stress': (['ky', 'mode', 'species', 'field'], ql_data[:, :, :, :, 3]),
        'exchange': (['ky', 'mode', 'species', 'field'], ql_data[:, :, :, :, 4]),
        'ky': (['ky'], ky_values),
    })


def load_tglf_input_params(batch_dir):
    """
    Load TGLF input parameters from input.tglf file.
    Returns a dictionary with all the parameters needed.
    """
    input_path = os.path.join(batch_dir, "input.tglf")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input.tglf in {batch_dir}")
    
    params = {}
    
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate type
                try:
                    if '.' in value or 'e' in value.lower() or 'E' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
    
    return params


def prepare_tglf_inputs(batch_dir, sat_rule, alpha_zf):
    """
    Prepare all inputs needed for the flux calculation.
    """
    params = load_tglf_input_params(batch_dir)
    
    in0 = {}
    
    # Required parameters
    in0['ALPHA_E'] = params.get('ALPHA_E', 1.0)
    in0['RLNP_CUTOFF'] = params.get('RLNP_CUTOFF', 18.0)
    in0['ALPHA_QUENCH'] = params.get('ALPHA_QUENCH', 0.0)
    in0['ALPHA_ZF'] = alpha_zf
    in0['TAUS_1'] = params.get('TAUS_1', 1.0)
    in0['AS_1'] = params.get('AS_1', 1.0)
    in0['SAT_RULE'] = sat_rule
    in0['UNITS'] = 'GYRO' if sat_rule == 1 else 'CGYRO'
    
    # Geometry and physics parameters
    for key in ['NS', 'RMAJ_LOC', 'RMIN_LOC', 'KAPPA_LOC', 'S_KAPPA_LOC',
                'DELTA_LOC', 'S_DELTA_LOC', 'ZETA_LOC', 'S_ZETA_LOC',
                'Q_LOC', 'Q_PRIME_LOC', 'P_PRIME_LOC', 'DRMAJDX_LOC',
                'DRMINDX_LOC', 'VEXB_SHEAR', 'SIGN_IT', 'BT_EXP',
                'USE_AVE_ION_GRID']:
        if key in params:
            in0[key] = params[key]
    
    # Species parameters
    in0['MASS_2'] = params.get('MASS_2', 2.0)
    in0['TAUS_2'] = params.get('TAUS_2', 1.0)
    in0['ZS_2'] = params.get('ZS_2', 1.0)
    
    # Handle multiple species
    if 'NS' in params:
        ns = int(params['NS'])
        for i in range(1, ns + 1):
            for prefix in ['MASS', 'TAUS', 'ZS', 'AS', 'RLNS', 'RLTS']:
                key = f'{prefix}_{i}'
                if key in params:
                    in0[key] = params[key]
    
    # Calculate rho_ion
    use_ave_ion = params.get('USE_AVE_ION_GRID', 0)
    
    if not use_ave_ion:
        in0['rho_ion'] = np.sqrt(in0['MASS_2'] * in0['TAUS_2']) / in0['ZS_2']
    else:
        rho_ion = 0.0
        charge = 0.0
        ns = int(in0.get('NS', 2))
        
        for is_ in range(1, ns + 1):
            zs_key = f'ZS_{is_}'
            as_key = f'AS_{is_}'
            mass_key = f'MASS_{is_}'
            taus_key = f'TAUS_{is_}'
            
            if all(k in in0 for k in [zs_key, as_key]):
                zs = in0[zs_key]
                as_val = in0[as_key]
                
                if is_ > 1 and (zs * as_val) / abs(in0['AS_1'] * in0.get('ZS_1', 1)) > 0.1:
                    charge += zs * as_val
                    if mass_key in in0 and taus_key in in0:
                        rho_ion += zs * as_val * np.sqrt(in0[mass_key] * in0[taus_key]) / zs
        
        in0['rho_ion'] = rho_ion / charge if charge > 0 else 1.0
    
    in0['SAT_geo0_out'] = 1.0
    
    return in0


def main():
    parser = argparse.ArgumentParser(
        description="Run TGLF flux calculation"
    )
    parser.add_argument(
        "--batch_dir", type=str, required=True,
        help="Path to the batch directory (e.g., ./batch-000)"
    )
    parser.add_argument(
        "--sat_rule", type=int, default=2,
        help="Saturation rule (1, 2, or 3)"
    )
    parser.add_argument(
        "--alpha_zf", type=float, default=1.0,
        help="Zonal flow coupling coefficient (default=1.0)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: <batch_dir>/tglf_flux_results.nc)"
    )
    
    args = parser.parse_args()
    batch_dir = os.path.abspath(args.batch_dir)

    # Check files exist
    growth_path = os.path.join(batch_dir, "out.tglf.eigenvalue_spectrum")
    ql_path = os.path.join(batch_dir, "out.tglf.QL_flux_spectrum")

    if not os.path.exists(growth_path):
        raise FileNotFoundError(f"Missing {growth_path}")
    if not os.path.exists(ql_path):
        raise FileNotFoundError(f"Missing {ql_path}")

    print(f"\n📂 Running TGLF flux calculation for: {batch_dir}")
    print(f"   Using SAT_RULE={args.sat_rule}, ALPHA_ZF={args.alpha_zf}\n")

    # Load TGLF output data
    print("Loading TGLF output files...")
    tglf_growth = load_tglf_eigenvalue_spectrum(growth_path)
    tglf_ql = load_tglf_ql_spectrum(ql_path)
    
    print(f"  Loaded {len(tglf_growth['ky'])} ky points")
    print(f"  ky range: {tglf_growth['ky'].values[0]:.3f} to {tglf_growth['ky'].values[-1]:.3f}")
    
    # Prepare inputs
    print("\nPreparing input parameters...")
    in0 = prepare_tglf_inputs(batch_dir, args.sat_rule, args.alpha_zf)
    
    # Extract growth rates - need to match the structure expected by the calculation
    tglf_kys = tglf_growth['ky'].values
    tglf_growth_rates = tglf_growth['gamma'].values
    
    # We need growth rates for all modes, not just the most unstable
    # For now, create a dummy array with the same growth rate for all modes
    # In reality, TGLF outputs multiple modes but we're only using the first
    nmodes = tglf_ql['particle_flux'].shape[1]  # Get number of modes from QL data
    tglf_gammas = np.zeros((len(tglf_kys), nmodes))
    tglf_gammas[:, 0] = tglf_growth_rates  # Most unstable mode
    # Other modes set to zero (subdominant)
    
    # Calculate saturation parameters
    print("Calculating saturation parameters...")
    # For sat params, we need the transposed version (modes x ky)
    gammas_for_sat = tglf_gammas.T
    kx0_e, satgeo1, satgeo2, R_unit, bt0, bgeo0, gradr0, _, _, _, _ = get_sat_params(
        args.sat_rule, tglf_kys, gammas_for_sat, **in0
    )
    
    # Update in0 with calculated geometry parameters
    in0['SAT_geo1_out'] = satgeo1
    in0['SAT_geo2_out'] = satgeo2
    in0['B_geo0_out'] = bgeo0
    in0['Bt0_out'] = bt0
    in0['grad_r0_out'] = gradr0
    
    # Extract QL weights from TGLF output
    particle_QL = tglf_ql['particle_flux'].values
    energy_QL = tglf_ql['energy_flux'].values
    toroidal_stress_QL = tglf_ql['toroidal_stress'].values
    parallel_stress_QL = tglf_ql['parallel_stress'].values
    exchange_QL = tglf_ql['exchange'].values
    
    # Run flux calculation
    print("Running flux calculation...")
    tglf_sat = sum_ky_spectrum(
        args.sat_rule,
        tglf_kys,
        tglf_gammas,  # Pass (ky, nmodes) array
        np.zeros_like(tglf_kys),
        R_unit,
        kx0_e,
        np.zeros((len(tglf_kys), nmodes)),  # Match shape
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
    
    print("\n✅ TGLF flux calculation complete.\n")
    print("Results summary (Gyro-Bohm normalized):")
    print(f"  Particle flux (Γ): {tglf_satG}")
    print(f"  Energy flux (Q):   {tglf_satQ}")
    print(f"  Momentum flux (Π): {tglf_satP}")
    
    # Save results
    output_path = args.output or os.path.join(batch_dir, "tglf_flux_results.nc")
    
    species_names = ['electron', 'ion1', 'ion2'][:len(tglf_satG)]
    
    result_ds = xr.Dataset({
        'particle_flux': (['species'], tglf_satG),
        'energy_flux': (['species'], tglf_satQ),
        'momentum_flux': (['species'], tglf_satP),
    }, coords={
        'species': species_names
    })
    
    result_ds.to_netcdf(output_path)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Save to HDF5 format as well
    h5_output_path = args.output or os.path.join(batch_dir, "tglf_flux_results.h5")
    if h5_output_path.endswith('.nc'):
        h5_output_path = h5_output_path.replace('.nc', '.h5')
    
    save_tglf_results_to_h5(
        h5_path=h5_output_path,
        batch_dir=batch_dir,
        sat_rule=args.sat_rule,
        alpha_zf=args.alpha_zf,
        fluxes=(tglf_satG, tglf_satQ, tglf_satP),
        ky_values=tglf_kys,
        sum_flux_spectrum=tglf_sat['sum_flux_spectrum'],
        growth_rates=tglf_growth_rates,
        in0=in0
    )
    print(f"💾 HDF5 results saved to: {h5_output_path}")


def save_tglf_results_to_h5(h5_path, batch_dir, sat_rule, alpha_zf, fluxes, 
                            ky_values, sum_flux_spectrum, growth_rates, in0):
    """
    Save TGLF flux calculation results to HDF5 file.
    This function is compatible with the batch processing workflow.
    
    Parameters:
    -----------
    h5_path : str
        Path to output HDF5 file
    batch_dir : str
        Path to batch directory (for batch index extraction)
    sat_rule : int
        Saturation rule used (1, 2, or 3)
    alpha_zf : float
        Zonal flow coefficient
    fluxes : tuple
        (particle_flux, energy_flux, momentum_flux) per species
    ky_values : array
        ky spectrum values
    sum_flux_spectrum : array
        Flux spectrum over ky (nky, nmodes, nspecies, nfield, ntype)
    growth_rates : array
        Growth rates per ky
    in0 : dict
        Input parameters dictionary
    """
    import h5py
    
    tglf_satG, tglf_satQ, tglf_satP = fluxes
    
    # Compute scalar fluxes following the convention: G_e, Q_e, Q_i, P_i
    G_elec = float(tglf_satG[0])
    Q_elec = float(tglf_satQ[0])
    Q_ions = float(np.sum(tglf_satQ[1:]))  # sum over all ion species
    P_ions = float(np.sum(tglf_satP[1:]))
    
    # Create output arrays
    fluxes_out = np.array([G_elec, Q_elec, Q_ions, P_ions], dtype=np.float32)
    
    # Sum flux spectrum over modes: (nky, nmodes, ns, nf, 5) -> (nky, ns, nf, 5)
    # sumf = np.sum(sum_flux_spectrum, axis=1).astype(np.float32)

    # keeping sumf as is (nky, nmodes, ns, nf, 5)
    sumf = sum_flux_spectrum.astype(np.float32)
    
    # Prepare ky array
    ky_array = ky_values.astype(np.float32)
    
    # Prepare input dictionary with TGLF keys
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
    
    # Append to HDF5 using the same structure as parse_outputs.py
    append_to_h5_individual_keys(
        h5_path=h5_path,
        input_dict=input_dict,
        fluxes=fluxes_out,
        sumf=sumf,
        ky=ky_array,
        meta={
            'sat_rule': sat_rule,
            'alpha_zf': alpha_zf,
            'nky': len(ky_array),
            'nspecies': len(tglf_satG),
        }
    )


def append_to_h5_individual_keys(h5_path, input_dict, fluxes, sumf, ky, meta=None):
    """
    Append data to HDF5 file using individual keys.
    Compatible with the batch processing workflow from parse_outputs.py
    """
    import h5py
    
    fluxes = np.atleast_1d(fluxes).astype(np.float32)
    sumf = np.asarray(sumf, dtype=np.float32)
    ky = np.asarray(ky, dtype=np.float32)
    
    with h5py.File(h5_path, 'a') as f:
        print("📥 Appending to HDF5:")
        print(f"  fluxes: {fluxes.shape}")
        print(f"  sumf: {sumf.shape}")
        print(f"  ky: {ky.shape}")
        
        # Save scalar input keys
        for name, data in input_dict.items():
            data = np.atleast_1d(data).astype(np.float32)
            if name not in f:
                f.create_dataset(name, data=data, maxshape=(None,), chunks=True)
            else:
                f[name].resize(f[name].shape[0] + 1, axis=0)
                f[name][-1] = data
        
        # Save flux vector (G_e, Q_e, Q_i, P_i)
        flux_names = ["OUT_G_e", "OUT_Q_e", "OUT_Q_i", "OUT_P_i"]
        for i, name in enumerate(flux_names):
            val = np.float32(fluxes[i])
            if name not in f:
                f.create_dataset(name, data=[val], maxshape=(None,), chunks=True)
            else:
                f[name].resize((f[name].shape[0] + 1), axis=0)
                f[name][-1] = val
        
        # Save sumf matrix: (1, nky, ns, nf, 5)
        if "sumf" not in f:
            f.create_dataset("sumf", data=sumf[None, ...], maxshape=(None,) + sumf.shape, chunks=True)
        else:
            f["sumf"].resize((f["sumf"].shape[0] + 1), axis=0)
            f["sumf"][-1] = sumf
        
        # Save ky array: (1, nky)
        if "ky" not in f:
            f.create_dataset("ky", data=ky[None, :], maxshape=(None, ky.shape[0]), chunks=True)
        else:
            f["ky"].resize((f["ky"].shape[0] + 1), axis=0)
            f["ky"][-1] = ky
        
        # Save meta info
        if meta:
            meta_grp = f.require_group("meta")
            for key, value in meta.items():
                value = np.asarray(value)
                if key not in meta_grp:
                    meta_grp.create_dataset(
                        key, 
                        data=value[None, ...] if value.ndim > 0 else value[None],
                        maxshape=(None,) + value.shape if value.ndim > 0 else (None,),
                        chunks=True
                    )
                else:
                    meta_grp[key].resize(meta_grp[key].shape[0] + 1, axis=0)
                    meta_grp[key][-1] = value

if __name__ == "__main__":
    main()