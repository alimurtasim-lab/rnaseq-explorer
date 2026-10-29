"""Headless run of the full analysis: writes tables + figures to results/.

    python scripts/run_pipeline.py \
        --counts data/demo_counts.csv --metadata data/demo_metadata.csv \
        --condition condition --reference Control --test Treated --covariates batch
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from rnaseq_explorer import de, io, static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="data/demo_counts.csv")
    ap.add_argument("--metadata", default="data/demo_metadata.csv")
    ap.add_argument("--condition", default="condition")
    ap.add_argument("--reference", default="Control")
    ap.add_argument("--test", default="Treated")
    ap.add_argument("--covariates", nargs="*", default=[])
    ap.add_argument("--padj", type=float, default=0.05)
    ap.add_argument("--lfc", type=float, default=1.0)
    ap.add_argument("--truth", default="data/demo_ground_truth.csv", help="optional ground-truth file for validation")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(exist_ok=True)
    counts, meta = io.validate(io.load_counts(a.counts), io.load_metadata(a.metadata), a.condition)
    r = de.run_deseq2(counts, meta, a.condition, a.reference, a.test, covariates=a.covariates)

    res = r.results.copy()
    res["category"] = de.classify(res, a.padj, a.lfc)
    res.sort_values("padj").to_csv(out / "de_results.csv")
    r.norm_counts.to_csv(out / "normalized_counts.csv")

    scores, var = de.pca(r.log_norm)
    top = res.dropna(subset=["padj"]).nsmallest(50, "padj").index.tolist()
    static.volcano_png(res, out / "volcano.png", a.padj, a.lfc)
    static.ma_png(res, out / "ma_plot.png", a.padj, a.lfc)
    static.pca_png(scores, var, r.metadata, a.condition, out / "pca.png")
    static.heatmap_png(r.log_norm, top, r.metadata, a.condition, out / "heatmap_top50.png")

    summary = {
        "genes_input": r.n_genes_input, "genes_tested": r.n_genes_tested,
        "n_up": int((res.category == "Up").sum()), "n_down": int((res.category == "Down").sum()),
        "padj_cutoff": a.padj, "abs_log2FC_cutoff": a.lfc,
        "pc1_var": round(float(var[0]), 3), "pc2_var": round(float(var[1]), 3),
    }

    if a.truth and Path(a.truth).exists():
        t = pd.read_csv(a.truth, index_col=0)
        j = res.join(t, how="inner")
        called = j.category != "NS"
        tp = int((called & j.is_de).sum()); fp = int((called & ~j.is_de).sum())
        fn = int((~called & j.is_de).sum())
        # sign agreement among true positives
        tpd = j[called & j.is_de]
        summary["validation"] = {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(tp / max(tp + fp, 1), 3),
            "recall": round(tp / max(tp + fn, 1), 3),
            "observed_FDR": round(fp / max(tp + fp, 1), 3),
            "lfc_sign_concordance": round(float((tpd.log2FoldChange * tpd.true_log2FC > 0).mean()), 3),
            "lfc_pearson_r_true_DE": round(float(tpd.log2FoldChange.corr(tpd.true_log2FC)), 3),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
