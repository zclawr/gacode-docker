#!/usr/bin/env python3
# plot_ky_histograms.py
import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def ensure_outdir(d):
    os.makedirs(d, exist_ok=True)
    return d

def load_ky(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "ky" not in f:
            raise KeyError("HDF5 file missing dataset 'ky'.")
        ky = f["ky"][:]  # shape (samples, nky_total)
    if ky.ndim != 2:
        raise ValueError(f"'ky' must be 2D, got shape {ky.shape}")
    return ky

def finite_only(x):
    m = np.isfinite(x)
    return x[m], m.sum()

def compute_stats(x):
    """Return dict of summary stats (nan-safe) for a 1D array x."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(count=0, mean=np.nan, std=np.nan, min=np.nan,
                    p5=np.nan, p50=np.nan, p95=np.nan, max=np.nan)
    return dict(
        count=int(x.size),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=0)),
        min=float(np.min(x)),
        p5=float(np.percentile(x, 5)),
        p50=float(np.percentile(x, 50)),
        p95=float(np.percentile(x, 95)),
        max=float(np.max(x)),
    )

def plot_hist(ax, data, bins, title):
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("k_y value (at this index across samples)")
    ax.set_ylabel("count")
    # annotate mean/median
    if data.size:
        mu = np.mean(data)
        med = np.median(data)
        ax.axvline(mu, linestyle="--", linewidth=1)
        ax.axvline(med, linestyle=":", linewidth=1)
        ax.legend([f"mean={mu:.4g}", f"median={med:.4g}"], frameon=False)

def main():
    ap = argparse.ArgumentParser(description="Generate per-index histograms for ky from an HDF5 file.")
    ap.add_argument("--h5", required=True, help="Path to HDF5 produced earlier (must contain dataset 'ky').")
    ap.add_argument("--outdir", default="ky_histograms", help="Directory to write PNGs/CSV/PDF.")
    ap.add_argument("--bins", type=int, default=50, help="Number of histogram bins (default: 50).")
    ap.add_argument("--pdf", action="store_true", help="Also write a combined multi-page PDF.")
    ap.add_argument("--prefix", default="ky_hist", help="Filename prefix for PNGs.")
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)
    ky = load_ky(args.h5)
    n_samples, nky = ky.shape
    print(f"Loaded ky with shape {ky.shape} (samples={n_samples}, nky={nky}).")

    stats_rows = []
    pdf_path = os.path.join(outdir, f"{args.prefix}.pdf") if args.pdf else None
    pdf = PdfPages(pdf_path) if args.pdf else None

    for j in range(nky):
        col = ky[:, j]
        col_finite, n_fin = finite_only(col)
        if n_fin == 0:
            print(f"[WARN] ky index {j}: all values NaN/Inf; skipping plot & stats.")
            continue

        # stats
        st = compute_stats(col_finite)
        st["ky_index"] = j
        stats_rows.append(st)

        # plot
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_hist(ax, col_finite, bins=args.bins, title=f"ky histogram @ index {j} (N={n_fin})")
        png_path = os.path.join(outdir, f"{args.prefix}_idx{j:02d}.png")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        if pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)

    # save stats CSV
    if stats_rows:
        import csv
        csv_path = os.path.join(outdir, f"{args.prefix}_stats.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ky_index","count","mean","std","min","p5","p50","p95","max"])
            w.writeheader()
            for st in stats_rows:
                w.writerow(st)
        print(f"Wrote stats: {csv_path}")
    else:
        print("No finite data found across indices; no stats CSV written.")

    if pdf is not None:
        pdf.close()
        print(f"Wrote PDF: {pdf_path}")

    print(f"Wrote PNGs to: {outdir}")

if __name__ == "__main__":
    main()
