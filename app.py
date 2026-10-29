"""Streamlit front-end for RNA-seq Explorer.  Run:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

from rnaseq_explorer import de, io, plots

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="RNA-seq Explorer", page_icon="🧬", layout="wide")
st.title("🧬 RNA-seq Explorer")
st.caption("Upload raw counts → PyDESeq2 differential expression → interactive volcano, PCA and heatmap.")


# ---------- cached computation ----------
@st.cache_data(show_spinner="Running PyDESeq2 (size factors → dispersions → Wald test)…")
def cached_de(counts, meta, cond, ref, test, covs, min_count, shrink):
    return de.run_deseq2(counts, meta, cond, ref, test, list(covs), min_count, shrink)


# ---------- sidebar: data ----------
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", ["Demo dataset (simulated)", "Upload my own"], label_visibility="collapsed")
    if source.startswith("Demo"):
        counts_raw = io.load_counts(DATA / "demo_counts.csv")
        meta_raw = io.load_metadata(DATA / "demo_metadata.csv")
    else:
        c_file = st.file_uploader("Count matrix (CSV/TSV, genes × samples)", type=["csv", "tsv", "txt"])
        m_file = st.file_uploader("Sample metadata (CSV/TSV)", type=["csv", "tsv", "txt"])
        if not (c_file and m_file):
            st.info("Upload both files, or switch to the demo dataset.")
            st.stop()
        try:
            counts_raw, meta_raw = io.load_counts(c_file), io.load_metadata(m_file)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read files: {e}")
            st.stop()

    st.header("2 · Design")
    cond_col = st.selectbox("Condition column", meta_raw.columns, index=0)
    levels = sorted(meta_raw[cond_col].astype(str).unique())
    ref = st.selectbox("Reference level", levels, index=0)
    test = st.selectbox("Test level", [l for l in levels if l != ref])
    covs = st.multiselect("Covariates (e.g. batch)", [c for c in meta_raw.columns if c != cond_col])
    min_count = st.number_input("Min total reads per gene", 0, 1000, 10)
    shrink = st.checkbox("Shrink log2FC (apeGLM)", value=False, help="Slower, but stabilises fold changes of low-count genes.")

    st.header("3 · Thresholds")
    padj_cut = st.slider("Adjusted p-value <", 0.001, 0.2, 0.05, 0.001, format="%.3f")
    lfc_cut = st.slider("|log2 fold change| ≥", 0.0, 4.0, 1.0, 0.1)

try:
    counts, meta = io.validate(counts_raw, meta_raw, cond_col)
except io.InputError as e:
    st.error(str(e))
    st.stop()

try:
    r = cached_de(counts, meta, cond_col, ref, test, tuple(covs), int(min_count), shrink)
except Exception as e:  # noqa: BLE001
    st.error(f"DESeq2 failed: {e}")
    st.stop()

res = r.results.copy()
res["category"] = de.classify(res, padj_cut, lfc_cut)
n_up, n_down = (res.category == "Up").sum(), (res.category == "Down").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Genes tested", f"{r.n_genes_tested:,}", f"{r.n_genes_input - r.n_genes_tested} filtered", delta_color="off")
c2.metric("Samples", len(r.metadata))
c3.metric("Up in " + test, int(n_up))
c4.metric("Down in " + test, int(n_down))

tab_volc, tab_ma, tab_pca, tab_heat, tab_gene, tab_table, tab_data = st.tabs(
    ["🌋 Volcano", "📈 MA plot", "🧭 PCA", "🔥 Heatmap", "🔎 Gene lookup", "📋 Results table", "🗂 Input data"]
)

# ---------- volcano with lasso/box selection ----------
with tab_volc:
    st.write("Hover for gene details. **Box- or lasso-select points** to list those genes below.")
    fig = plots.volcano(res, padj_cut, lfc_cut)
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=("box", "lasso", "points"), key="volcano")
    sel_genes = []
    if ev and ev.selection and ev.selection.points:
        sel_genes = sorted({p.get("hovertext") or p.get("hover_name") for p in ev.selection.points if p.get("hovertext") or p.get("hover_name")})
    if sel_genes:
        st.success(f"{len(sel_genes)} genes selected")
        st.dataframe(res.loc[res.index.intersection(sel_genes), ["baseMean", "log2FoldChange", "padj", "category"]]
                     .sort_values("padj"), use_container_width=True)
        st.download_button("Download selection (CSV)", res.loc[res.index.intersection(sel_genes)].to_csv(), "selected_genes.csv")

with tab_ma:
    st.plotly_chart(plots.ma_plot(res, padj_cut, lfc_cut), use_container_width=True)

with tab_pca:
    k1, k2, k3 = st.columns(3)
    n_top = k1.slider("Top variable genes", 50, min(5000, len(r.log_norm)), min(500, len(r.log_norm)), 50)
    color_by = k2.selectbox("Colour by", r.metadata.columns, index=list(r.metadata.columns).index(cond_col))
    symbol_by = k3.selectbox("Shape by", [None] + list(r.metadata.columns))
    scores, var = de.pca(r.log_norm, n_top)
    st.plotly_chart(plots.pca_plot(scores, var, r.metadata, color_by, symbol_by), use_container_width=True)
    st.caption("Samples should cluster by biology, not by technical covariates. If they separate by batch, include it as a covariate.")

with tab_heat:
    h1, h2 = st.columns(2)
    n_heat = h1.slider("Number of genes", 5, 100, 40)
    mode = h2.radio("Pick genes by", ["Smallest adj. p-value", "Largest |log2FC| (significant)"], horizontal=True)
    d = res.dropna(subset=["padj"])
    if mode.startswith("Smallest"):
        genes = d.nsmallest(n_heat, "padj").index.tolist()
    else:
        sig = d[d.category != "NS"]
        genes = sig.reindex(sig.log2FoldChange.abs().sort_values(ascending=False).index).head(n_heat).index.tolist()
    if len(genes) < 3:
        st.warning("Fewer than 3 genes pass the thresholds; relax them in the sidebar.")
    else:
        st.plotly_chart(plots.heatmap(r.log_norm, genes, r.metadata, cond_col), use_container_width=True)

with tab_gene:
    q = st.selectbox("Search a gene", res.sort_values("padj").index, index=0)
    g1, g2 = st.columns([1, 1])
    g1.plotly_chart(plots.gene_boxplot(r.log_norm, q, r.metadata, cond_col), use_container_width=True)
    row = res.loc[q]
    g2.markdown(f"**{q}**")
    g2.table(pd.DataFrame({"value": [f"{row.baseMean:.1f}", f"{row.log2FoldChange:.2f}", f"{row.pvalue:.2e}", f"{row.padj:.2e}", row.category]},
                          index=["Mean normalized count", "log2 fold change", "p-value", "adj. p-value", "Call"]))

with tab_table:
    show = st.radio("Show", ["All", "Significant only", "Up", "Down"], horizontal=True)
    t = res if show == "All" else res[res.category != "NS"] if show.startswith("Sig") else res[res.category == show]
    t = t.sort_values("padj")[["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj", "category"]]
    st.dataframe(t, use_container_width=True, height=450)
    st.download_button("Download results (CSV)", t.to_csv(), f"deseq2_{test}_vs_{ref}.csv")
    st.download_button("Download normalized counts (CSV)", r.norm_counts.to_csv(), "normalized_counts.csv")

with tab_data:
    st.write("**Design formula:** `~ " + " + ".join([cond_col] + covs) + f"`  ·  contrast: **{test} vs {ref}**")
    st.dataframe(meta, use_container_width=True)
    st.dataframe(counts.head(100), use_container_width=True)
    if counts.attrs.get("dropped_samples"):
        st.warning(f"Samples present in only one file were dropped: {counts.attrs['dropped_samples']}")

st.divider()
st.caption("Built with PyDESeq2 (Muzellec et al., Bioinformatics 2023), Streamlit and Plotly.")
