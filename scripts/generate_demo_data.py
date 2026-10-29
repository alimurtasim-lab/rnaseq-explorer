"""Generate a reproducible *simulated* bulk RNA-seq dataset with known ground truth.

Why simulated? Because we know exactly which genes were spiked as differentially
expressed, so we can measure how well the pipeline recovers them (see README,
"Validation"). The model is a negative binomial with gene-specific baseline
means and dispersions, library-size variation and a mild batch effect.

Usage:
    python scripts/generate_demo_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_GENES = 2000
N_PER_GROUP = 6
FRAC_DE = 0.10  # 200 truly DE genes (100 up, 100 down)

rng = np.random.default_rng(SEED)
out = Path(__file__).resolve().parents[1] / "data"
out.mkdir(exist_ok=True)

genes = [f"GENE{str(i).zfill(4)}" for i in range(1, N_GENES + 1)]
samples = [f"Control_{i}" for i in range(1, N_PER_GROUP + 1)] + [
    f"Treated_{i}" for i in range(1, N_PER_GROUP + 1)
]
condition = ["Control"] * N_PER_GROUP + ["Treated"] * N_PER_GROUP
batch = ["A", "B"] * N_PER_GROUP  # balanced across conditions

# Gene-level baseline mean (log-normal) and dispersion (mean-dependent trend + noise)
base_mean = rng.lognormal(mean=5.0, sigma=1.5, size=N_GENES)
dispersion = 0.1 + 1.0 / np.sqrt(base_mean) + rng.normal(0, 0.03, N_GENES)
dispersion = np.clip(dispersion, 0.02, None)

# Ground truth log2 fold changes
n_de = int(N_GENES * FRAC_DE)
de_idx = rng.choice(N_GENES, size=n_de, replace=False)
lfc = np.zeros(N_GENES)
signs = np.array([1] * (n_de // 2) + [-1] * (n_de - n_de // 2))
rng.shuffle(signs)
lfc[de_idx] = signs * rng.uniform(1.0, 3.0, size=n_de)

# Sample-level size factors and mild batch effect on a random 15% of genes
size_factors = rng.uniform(0.7, 1.4, size=len(samples))
batch_genes = rng.random(N_GENES) < 0.15
batch_lfc = np.where(batch_genes, rng.normal(0, 0.4, N_GENES), 0.0)

counts = np.zeros((N_GENES, len(samples)), dtype=int)
for j, (cond, b) in enumerate(zip(condition, batch)):
    log2_mu = np.log2(base_mean)
    if cond == "Treated":
        log2_mu = log2_mu + lfc
    if b == "B":
        log2_mu = log2_mu + batch_lfc
    mu = (2 ** log2_mu) * size_factors[j]
    r = 1.0 / dispersion  # NB size parameter
    p = r / (r + mu)
    counts[:, j] = rng.negative_binomial(r, p)

pd.DataFrame(counts, index=genes, columns=samples).rename_axis("gene_id").to_csv(
    out / "demo_counts.csv"
)
pd.DataFrame({"condition": condition, "batch": batch}, index=samples).rename_axis(
    "sample"
).to_csv(out / "demo_metadata.csv")
pd.DataFrame(
    {"gene_id": genes, "true_log2FC": lfc, "is_de": lfc != 0}
).to_csv(out / "demo_ground_truth.csv", index=False)

print(f"Wrote {N_GENES} genes x {len(samples)} samples to {out}")
print(f"Truly DE genes: {n_de} (up={int((lfc>0).sum())}, down={int((lfc<0).sum())})")
