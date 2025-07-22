import os
import h5py
from datetime import datetime
from pyrokinetics import Pyro
import numpy as np
import textwrap

input_keys = [
    "RLTS_3", "KAPPA_LOC", "ZETA_LOC", "TAUS_3", "VPAR_1", "Q_LOC", "RLNS_1",
    "TAUS_2", "Q_PRIME_LOC", "P_PRIME_LOC", "ZMAJ_LOC", "VPAR_SHEAR_1",
    "RLTS_2", "S_DELTA_LOC", "RLTS_1", "RMIN_LOC", "DRMAJDX_LOC", "AS_3",
    "RLNS_3", "DZMAJDX_LOC", "DELTA_LOC", "S_KAPPA_LOC", "ZEFF", "VEXB_SHEAR",
    "RMAJ_LOC", "AS_2", "RLNS_2", "S_ZETA_LOC", "BETAE_log10", "XNUE_log10", "DEBYE_log10"
]
log_ops_keys = {"BETAE_log10", "XNUE_log10", "DEBYE_log10"}

FIXED_TRAILER_TEMPLATE = textwrap.dedent("""\
    GEOMETRY_FLAG = 1
    SIGN_BT=-1.00000E+00
    SIGN_IT=+1.00000E+00

    #----------Additional Parameters----------
    # Species
    NS=3
    N_MODES=5
    # Questionable forced defaults:

    NKY=1
    USE_BPER=True
    USE_BPAR=True
    USE_AVE_ION_GRID=True
    USE_MHD_RULE=False
    ALPHA_ZF=-1
    KYGRID_MODEL=0
    KY={ky_val}
    SAT_RULE=3
    NBASIS_MAX=6
    UNITS=CGYRO
    VPAR_2 = 0.0
    VPAR_3 = 0.0

    VPAR_SHEAR_2 = 0.0
    VPAR_SHEAR_3 = 0.0

    #Confirmed with Tom 7/19
    AS_1=+1.0
    TAUS_1=+1.0
    MASS_1=0.0002723125672605524
    ZS_1=-1
    MASS_2=+1.0
    ZS_2=1
    MASS_3=+6.0
    ZS_3=6.0
""")

def write_input_tglf(f, sample_idx, ky_idx, out_path):
    with open(out_path, "w") as f_out:
        f_out.write("# Geometry (Miller) and Parameters\n")
        for key in input_keys:
            val = f[key][sample_idx]
            if key in log_ops_keys:
                val = 10 ** val
                key_out = key.replace("_log10", "")
            else:
                key_out = key
            f_out.write(f"{key_out}={val:+.5E}\n")

        ky_val = f["ky"][sample_idx, ky_idx]
        trailer = FIXED_TRAILER_TEMPLATE.format(ky_val=f"{ky_val:+.5E}")
        f_out.write("\n" + trailer + "\n")

def generate_tglf_and_cgyro(f, sample_idx, ky_idx, tglf_out_path, cgyro_out_path):
    write_input_tglf(f, sample_idx, ky_idx, tglf_out_path)
    pyro = Pyro(gk_file=tglf_out_path)
    for species in pyro.local_species['names']:
        pyro.local_species.enforce_quasineutrality(species)
        break
    pyro.write_gk_file(cgyro_out_path, gk_code="CGYRO", enforce_quasineutrality=True)

def convert_h5_to_batch_dir(h5_path, out_root="all_batches"):
    os.makedirs(out_root, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n_samples, n_ky = f["ky"].shape

        for sample_idx in range(n_samples):
            batch_name = f"batch-{sample_idx:03d}"
            batch_dir = os.path.join(out_root, batch_name)
            tglf_base = os.path.join(batch_dir, "tglf")
            cgyro_base = os.path.join(batch_dir, "cgyro")

            os.makedirs(tglf_base, exist_ok=True)
            os.makedirs(cgyro_base, exist_ok=True)

            for ky_idx in range(n_ky):
                input_name = f"input-{ky_idx:03d}"
                tglf_dir = os.path.join(tglf_base, input_name)
                cgyro_dir = os.path.join(cgyro_base, input_name)

                os.makedirs(tglf_dir, exist_ok=True)
                os.makedirs(cgyro_dir, exist_ok=True)

                tglf_out_path = os.path.join(tglf_dir, "input.tglf")
                cgyro_out_path = os.path.join(cgyro_dir, "input.cgyro")

                generate_tglf_and_cgyro(f, sample_idx, ky_idx, tglf_out_path, cgyro_out_path)

    print(f"✅ Done. TGLF/CGYRO inputs written to subdirs in: {out_root}")



# === Run ===
h5_file = "./input_generation/sampled_output_file.h5"  # Replace with your actual file path
output_dir = "./cgyro_inputs"
convert_h5_to_batch_dir(h5_file, output_dir)
