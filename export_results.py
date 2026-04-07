"""
export_results.py — Aggregate and compare all experiment results.

Reads every results/<experiment>/metrics.json file, then produces:
  • results/comparison_table.csv    — FID, DACID, final losses for each run
  • results/comparison_metrics.png  — side-by-side FID & DACID bar charts
  • results/comparison_losses.png   — loss curves for all AttGAN experiments

Run from the repo root after training all experiments:
    python export_results.py

If some experiments haven't been run yet, they are simply skipped.
"""

import json
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT         = Path(__file__).parent.resolve()
RESULTS_ROOT = ROOT / "results"
EXPORT_DIR   = RESULTS_ROOT          # outputs land in results/ root

# Expected experiments in display order
EXPERIMENTS = [
    "simple_gan",
    "exp1_baseline",
    "exp2_high_rec",
    "exp3_strong_attr",
]

# Human-readable labels and short descriptions for charts/tables
LABELS = {
    "simple_gan":      "Simple GAN\n(DCGAN, unconditional)",
    "exp1_baseline":   "AttGAN — Exp 1\nBaseline (λ_rec=100)",
    "exp2_high_rec":   "AttGAN — Exp 2\nHigh rec (λ_rec=200)",
    "exp3_strong_attr":"AttGAN — Exp 3\nStrong attr (λ_rec=50, λ_cls_G=5)",
}

COLORS = {
    "simple_gan":      "#7F77DD",
    "exp1_baseline":   "#1D9E75",
    "exp2_high_rec":   "#BA7517",
    "exp3_strong_attr":"#D85A30",
}


# ─────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    """
    Scan results/ subfolders for metrics.json files.
    Returns a list of dicts, one per found experiment, in EXPERIMENTS order.
    """
    rows = []
    for exp in EXPERIMENTS:
        path = RESULTS_ROOT / exp / "metrics.json"
        if not path.exists():
            print(f"[export] Skipping {exp} — no metrics.json found")
            continue
        with open(path) as f:
            data = json.load(f)
        data["_exp_key"] = exp
        rows.append(data)
        print(f"[export] Loaded {exp}  FID={data.get('fid')}  "
              f"DACID={data.get('dacid')}")
    return rows


# ─────────────────────────────────────────────────────────────────────
# Export: CSV table
# ─────────────────────────────────────────────────────────────────────

def export_csv(rows: list[dict]):
    out = EXPORT_DIR / "comparison_table.csv"
    fieldnames = [
        "experiment", "model",
        "fid", "dacid",
        "final_g_loss", "final_d_loss",
        "n_epochs",
    ]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            g_losses = r.get("g_losses", [])
            d_losses = r.get("d_losses", [])
            w.writerow({
                "experiment":   r.get("experiment", r["_exp_key"]),
                "model":        r.get("model", "AttGAN"),
                "fid":          r.get("fid"),
                "dacid":        r.get("dacid"),
                "final_g_loss": round(g_losses[-1], 4) if g_losses else None,
                "final_d_loss": round(d_losses[-1], 4) if d_losses else None,
                "n_epochs":     len(g_losses),
            })
    print(f"[export] CSV  saved → {out}")


# ─────────────────────────────────────────────────────────────────────
# Export: Metrics bar chart (FID + DACID)
# ─────────────────────────────────────────────────────────────────────

def export_metrics_chart(rows: list[dict]):
    valid = [r for r in rows if r.get("fid") is not None
                              and r.get("dacid") is not None]
    if not valid:
        print("[export] No metric values found — skipping metrics chart")
        return

    keys   = [r["_exp_key"] for r in valid]
    labels = [LABELS.get(k, k).replace("\n", "\n") for k in keys]
    fids   = [r["fid"]   for r in valid]
    dacids = [r["dacid"] for r in valid]
    colors = [COLORS.get(k, "#888") for k in keys]

    x      = np.arange(len(keys))
    width  = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # FID
    bars1 = ax1.bar(x, fids, width*2, color=colors, edgecolor="white",
                    linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("FID (lower = better)")
    ax1.set_title("Fréchet Inception Distance", fontweight="bold")
    ax1.bar_label(bars1, fmt="%.1f", fontsize=8, padding=3)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(fids) * 1.2)

    # DACID
    bars2 = ax2.bar(x, dacids, width*2, color=colors, edgecolor="white",
                    linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("DACID (lower = better)")
    ax2.set_title("DACID Score", fontweight="bold")
    ax2.bar_label(bars2, fmt="%.1f", fontsize=8, padding=3)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(dacids) * 1.2)

    # Shared legend
    patches = [mpatches.Patch(color=COLORS.get(k, "#888"),
                               label=LABELS.get(k, k).split("\n")[0])
               for k in keys]
    fig.legend(handles=patches, loc="lower center", ncol=len(keys),
               fontsize=8, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle("AttGAN Experiment Comparison — FID & DACID",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = EXPORT_DIR / "comparison_metrics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[export] Metrics chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────
# Export: Loss curves for AttGAN experiments only
# ─────────────────────────────────────────────────────────────────────

def export_loss_curves(rows: list[dict]):
    attgan_rows = [r for r in rows if r["_exp_key"] != "simple_gan"
                   and r.get("g_losses")]
    if not attgan_rows:
        print("[export] No AttGAN loss data — skipping loss chart")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    for r in attgan_rows:
        k   = r["_exp_key"]
        lbl = LABELS.get(k, k).split("\n")[0]
        col = COLORS.get(k, "#888")
        ax1.plot(r["g_losses"], label=lbl, color=col)
        ax2.plot(r["d_losses"], label=lbl, color=col)

    ax1.set_title("Generator Loss");     ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=8);              ax1.grid(alpha=0.3)
    ax2.set_title("Discriminator Loss"); ax2.set_xlabel("Epoch")
    ax2.legend(fontsize=8);              ax2.grid(alpha=0.3)

    plt.suptitle("AttGAN Training Dynamics — All Experiments",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = EXPORT_DIR / "comparison_losses.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[export] Loss chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────
# Print summary table to stdout
# ─────────────────────────────────────────────────────────────────────

def print_summary(rows: list[dict]):
    print("\n" + "=" * 72)
    print(f"  {'Experiment':<24} {'Model':<12} {'FID':>8} {'DACID':>8} "
          f"{'G_loss':>9} {'D_loss':>9}")
    print("=" * 72)
    for r in rows:
        g = r.get("g_losses", [])
        d = r.get("d_losses", [])
        print(
            f"  {r['_exp_key']:<24} "
            f"{r.get('model','AttGAN'):<12} "
            f"{str(r.get('fid',  'n/a')):>8} "
            f"{str(r.get('dacid','n/a')):>8} "
            f"{(round(g[-1],4) if g else 'n/a'):>9} "
            f"{(round(d[-1],4) if d else 'n/a'):>9}"
        )
    print("=" * 72 + "\n")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results()

    if not rows:
        print("\n[export] No results found. Run at least one training script first.")
        return

    print_summary(rows)
    export_csv(rows)
    export_metrics_chart(rows)
    export_loss_curves(rows)

    print(f"\n[export] All exports saved to {EXPORT_DIR}/")
    print("  comparison_table.csv")
    print("  comparison_metrics.png")
    print("  comparison_losses.png")


if __name__ == "__main__":
    main()
