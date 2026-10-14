
from __future__ import annotations

import sys
import time
import io
import math
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy.stats import f_oneway, ttest_ind, spearmanr, norm
from statsmodels.stats.multitest import fdrcorrection, multipletests

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.decomposition import PCA
from adjustText import adjust_text
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from PIL import Image
import gseapy as gp

np.long = np.int64  # numpy/pandas compat shim used by the CIBERSORTx-merge stages
Image.MAX_IMAGE_PIXELS = None  # one SFigure7 panel PNG is ~179M pixels


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"   # all input files
# Final output: the composite manuscript figures.
FIGURE_DIR = SCRIPT_DIR / "2026_figure_plot"
# The individual panels are rendered to PNG in memory only (never written to
# disk) and assembled into the figures by stage_06. Panels are addressed by a
# virtual path under PANEL_ROOT, e.g. "panels/2026_ccrcc_subtype/....png".
PANEL_ROOT = Path("panels")
RESULT_MANUSCRIPT_DIR = PANEL_ROOT
PANELS = {}


def _panel_key(path):
    return Path(path).as_posix().lower()


def save_panel(path, fig=None, **savefig_kwargs):
    """Render the current (or given) figure as a PNG into memory under `path`."""
    buf = io.BytesIO()
    (fig if fig is not None else plt).savefig(buf, format='png', **savefig_kwargs)
    PANELS[_panel_key(path)] = buf.getvalue()


def open_panel(path):
    """Open a panel rendered earlier in this run by `save_panel`."""
    return Image.open(io.BytesIO(PANELS[_panel_key(path)]))


def read_gene_list(filename):
    """Read a gene-list CSV from `data/` (gene symbols kept verbatim, e.g. "NA")."""
    return pd.read_csv(DATA_DIR / filename, keep_default_na=False, na_values=[''])


def _reset_mpl_defaults():
    """Start every stage from matplotlib's library-default rcParams, so one
    stage's style settings (e.g. stage_03's 40-pt fonts) never carry over
    into the next stage's plots."""
    plt.close('all')
    mpl.rcParams.update(mpl.rcParamsDefault)


class _RwFunctionNamespace:
    """Small helpers shared by several stages (group colours, column lookup,
    density-coloured scatter), called as `rw_function.<name>(...)`."""

    @staticmethod
    def col_type(cancer = True):
        if cancer:
            return 'cluster_type'
        else:
            return 'Type'

    @staticmethod
    def desired_order(cancer =True):
        if cancer:
            return ['I', 'II']
        else:
            return ['Normal', 'Cancer']

    @staticmethod
    def find_col(df, base_name):
        if base_name in df.columns:
            return base_name
        elif f'{base_name}_x' in df.columns:
            return f'{base_name}_x' 
        elif f"{base_name}_y" in df.columns:
            return f"{base_name}_y"

    @staticmethod
    def scatterplot_kde_norm(x, y, bw='scott', s=.5, vmin=None, vmax=None, ax=None,
                             min_kde_points=10, use_kde=True, **kwargs):
        """Scatter plot coloured by point density (Gaussian KDE)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # keep only finite paired values
        valid = np.isfinite(x) & np.isfinite(y)
        x_plot = x[valid]
        y_plot = y[valid]

        if ax is None:
            ax = plt.gca()

        # default: no density coloring
        values = np.ones(len(x_plot))

        # only attempt KDE if enough points
        if use_kde and len(x_plot) >= min_kde_points:
            xy = np.vstack((x_plot, y_plot))
            try:
                kernel = sp.stats.gaussian_kde(xy, bw_method=bw)
                values = kernel(xy)
            except (np.linalg.LinAlgError, ValueError):
                values = np.ones(len(x_plot))

        if vmin is None:
            vmin = np.min(values) if len(values) else 0
        if vmax is None:
            vmax = np.max(values) if len(values) else 1

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis

        ax.scatter(x_plot, y_plot, c=cmap(norm(values)), s=s, **kwargs)
        return ax

rw_function = _RwFunctionNamespace()


# =============================================================================
# stage_01_ccrcc_subtype
# =============================================================================

def stage_01_ccrcc_subtype():
    """ccRCC glycosylation-based molecular subtyping (Normal vs. Cancer and
    Subtype I vs. II). Panels are rendered in memory.

    Pipeline summary
    -----------------
    1.  Load raw per-gene TPM/count data, attach sample metadata and clinical
        annotation, restrict to Kidney tissue, and filter to expressed genes
        (mitochondrial genes are kept).
    2.  Build the ccRCC (clear-cell RCC) sample set: all Normal samples plus
        all ccRCC tumours with clinical annotation.
    3.  Build glycoprotein / glycogene-pathway gene lists used throughout.
    4.  Molecularly subtype ccRCC tumors into two glycosylation clusters
        ("I" / "II") via Ward hierarchical clustering on z-scored glycogene
        expression (`cancer_glyco_clust`) -> clustermap (Figure3 A).
    5.  Test whether overall survival differs between cluster I and II
        (Kaplan-Meier + log-rank test).
    6.  Score which glycosylation pathway classes differ between groups
        (one-way ANOVA on per-sample pathway means) and plot the top pathways
        as a sorted, annotated z-score heatmap.
    7.  PCA of samples on the glycogenes of each top pathway, testing
        whether PCs separate the groups.
    8.  Per-gene Welch t-tests (mean-level and per-sample z-scored) between
        groups, plotted as fold-change/-log10(p) dot plots; the per-sample
        z-scores are passed on to stage_04.
    9.  Merge the CIBERSORTx immune-fraction estimates with the subtype/Type
        labels, plot them and pass them on to stage_04.
    10. Pass the cluster-annotated expression tables on to stages 02 and 03.
    11. Gene-of-interest (e.g. SIGLEC) and custom gene-list (immune-regulated
        genes) expression heatmaps comparing groups.

    Returns, in memory, the tables the later stages need (cluster-annotated
    expression, glycoprotein gene list, immune fractions, per-sample gene
    scores). The long-format cluster-annotated tables are trimmed to the
    columns later stages use (`trim_cluster_table`).
    """
    _reset_mpl_defaults()

    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import scipy as sp
    import scipy.stats as stats
    import scipy.cluster.hierarchy as sch
    from scipy.cluster.hierarchy import fcluster
    from scipy.stats import f_oneway, ttest_ind
    from statsmodels.stats.multitest import fdrcorrection

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns

    from sklearn.decomposition import PCA

    from adjustText import adjust_text

    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    SCRIPT_NAME = "2026_ccrcc_subtype"

    OUTPUT_DIR = RESULT_MANUSCRIPT_DIR / SCRIPT_NAME
    SAVE_DIR = OUTPUT_DIR


    # -----------------------------------------------------------------------------
    # General plotting helpers (not called by the pipeline below; available for
    # ad-hoc plots)
    # -----------------------------------------------------------------------------

    class LoggedDataLocator(mpl.ticker.Locator):
        """Set exponential tick marks for data that has already been logged."""

        def __init__(self, base=10, skip_base=False):
            self._base = base
            self._skip_base = skip_base

        def set_params(self, base):
            if base is not None:
                self._base = base

        def __call__(self):
            vmin, vmax = self.axis.get_view_interval()
            return self.tick_values(vmin, vmax)

        def tick_values(self, vmin, vmax):
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            locs = np.array(
                [
                    self._base ** c * range((2 if self._skip_base else 1), self._base)
                    for c in np.arange(np.floor(vmin // 1), np.ceil(vmax / 1))
                ]
            ).flatten()
            locs = locs[(locs > self._base ** vmin) & (locs < self._base ** vmax)]
            return np.log(locs) / np.log(self._base)

        def view_limits(self, dmin, dmax):
            return (
                self._base ** (dmin // 1) * np.floor(self._base ** ((dmin % 1))),
                self._base ** (dmax // 1) * np.ceil(self._base ** ((dmax % 1))),
            )


    # -----------------------------------------------------------------------------
    # Data loading, QC and gene filtering
    # -----------------------------------------------------------------------------

    def load_raw_expression():
        """Load the per-gene TPM/count table and add a log10(1+TPM) column."""
        df_raw = pd.read_csv(DATA_DIR / 'dataframe.tsv', sep='\t').reset_index(drop=True)
        df_raw = df_raw.drop(columns=df_raw.columns[0])
        df_raw['log_tpm'] = np.log10(1 + df_raw['transcripts_per_million'])
        return df_raw


    def load_sample_metadata():
        """Load the RNA-seq sample sheet and derive the integer `sample_id` key."""
        df_sample = pd.read_csv(DATA_DIR / 'samples for RNA seq.csv')
        df_sample['sample_id'] = df_sample['Count']
        return df_sample


    def merge_sample_metadata(df_raw, df_sample):
        """Attach sample metadata (patient/tissue/type info) to the expression table."""
        df_raw = df_raw.copy()
        df_raw['sample_id'] = df_raw['sample'].str.strip('AB').astype(int)
        return pd.merge(df_raw, df_sample, on='sample_id')


    def subset_kidney_tissue(df):
        return df[df['Tissue'] == 'Kidney']


    def filter_expressed_genes(df_kidney, min_samples=3, min_total_counts=500):
        """Low-expression filter on raw counts, then normalize. Mitochondrial
        genes are never removed -- they're treated like any other gene and
        are subject to the same low-expression filter (which they typically
        pass easily, being highly expressed) and contribute to the
        per-sample normalization total like everything else.

        Low-expression filter (on raw, pre-normalization counts): keep genes
        detected (counts > 0) in >= `min_samples` samples and with
        >= `min_total_counts` total counts across samples. Normalization
        (`transcripts_per_million`, `log_tpm`) is then recomputed on the
        filtered gene set, so it reflects that set's own per-sample library
        sizes rather than the original unfiltered totals
        (`transcripts_per_million` as loaded in `load_raw_expression` was
        computed across all genes, before this filtering).
        """
        df1 = df_kidney.copy()

        detected_keep = df1.groupby('ensembl')['counts'].apply(lambda x: (x > 0).sum() >= min_samples)
        count_keep = df1.groupby('ensembl')['counts'].sum() >= min_total_counts
        keep_ensembl = detected_keep.index[detected_keep & count_keep]

        df1 = df1[df1['ensembl'].isin(keep_ensembl)].copy()

        df1['transcripts_per_million'] = df1.groupby('sample')['counts'].transform(lambda x: 1e6 * x / x.sum())
        df1['log_tpm'] = np.log10(1 + df1['transcripts_per_million'])

        return df1


    # -----------------------------------------------------------------------------
    # Clinical annotation & ccRCC subsetting
    # -----------------------------------------------------------------------------

    def load_clinical_data():
        """Load clinical annotation and isolate the clear-cell RCC (ccRCC) patients."""
        df_clin = pd.read_csv(DATA_DIR / 'clin_data.csv')
        df_clin['patient_id'] = df_clin['Abbreviation'].astype(str).str.strip().str[:-1]
        df_clin = df_clin.rename(columns={'Type': 'cancer_type'})
        df_clin['cancer_type'] = df_clin['cancer_type'].astype(str).str.lower()

        df_ccrcc = df_clin[df_clin['cancer_type'].str.lower().str.contains('clear', na=False)]
        print(f'original samples: {df_clin.shape}, ccrcc samples: {df_ccrcc.shape}')
        return df_clin, df_ccrcc


    def df_cancer_type(df_kidney_orig, df_clin, df_ccrcc, match=True, ccRcc=True):
        """Build a Normal(+matched)/Cancer sample set annotated with clinical data.

        `match=True` keeps only cancer samples whose patient also has a normal
        sample; `ccRcc=True` further restricts to clear-cell RCC patients.
        """
        df_na = df_kidney_orig[df_kidney_orig['Type'] == 'Normal']
        df_ca = df_kidney_orig[df_kidney_orig['Type'] == 'Cancer']

        samples = df_na["Abbreviation"].str.strip().str[:-1].unique()
        df_ca_sub = df_ca[df_ca["Abbreviation"].str.strip().str[:-1].isin(samples)].copy()

        if match:
            df_k = pd.concat([df_na, df_ca_sub], axis=0, ignore_index=True)
        else:
            df_k = pd.concat([df_na, df_ca], axis=0, ignore_index=True)

        df_k['patient_id'] = df_k['Abbreviation'].str[:-1]
        df_k2 = df_k.merge(df_clin, on='patient_id', how='inner')
        if ccRcc:
            df_k2 = df_k2.merge(df_ccrcc, on='patient_id', how='inner')

        n = len(df_k2['patient_id'].unique())
        print(f'cancer samples: {n}')
        return df_k2


    # -----------------------------------------------------------------------------
    # Glycoprotein / glycogene-pathway gene lists
    # -----------------------------------------------------------------------------

    def build_glycoprotein_gene_list():
        """N-/O-glycoprotein gene list (used by stage_02)."""
        glycoprotein_genes = read_gene_list('glycoprotein_list.csv')['gene'].to_list()
        print(f"glycoprotein genes: {len(glycoprotein_genes)}")

        return glycoprotein_genes


    def load_pathway_class_table():
        """Load the gene -> glycosylation-pathway-class annotation table."""
        df_gly_pathway = read_gene_list('glycogene_list_380.csv')[['gene', 'pathway class']]
        gly_pathway_genes = df_gly_pathway['gene'].unique().tolist()
        return df_gly_pathway, gly_pathway_genes


    # -----------------------------------------------------------------------------
    # Small column-name helpers
    # -----------------------------------------------------------------------------

    def find_col(df, base_name):
        """Resolve a column name that may have been suffixed `_x`/`_y` by a merge."""
        if base_name in df.columns:
            return base_name
        elif f'{base_name}_x' in df.columns:
            return f'{base_name}_x'
        elif f'{base_name}_y' in df.columns:
            return f'{base_name}_y'


    def find_cols(clin_cols_ori, df_ca_sub_clin2):
        return [find_col(df_ca_sub_clin2, name) for name in clin_cols_ori]


    def find_col_name(cancer):
        return 'cluster_type' if cancer else 'Type'


    def desired_order(cancer):
        return ['I', 'II'] if cancer else ['Normal', 'Cancer']


    # -----------------------------------------------------------------------------
    # ccRCC molecular subtyping (glycosylation-based clustering) + survival
    # -----------------------------------------------------------------------------

    def cancer_glyco_clust(df_kidney, gly_pathway_genes, title, cancer_only=True, save_dir=SAVE_DIR,
                           plot=True):
        """Ward-cluster samples on z-scored glycogene-pathway expression into 2 groups.

        Plots a clustermap (samples colored red/blue by assigned cluster "I"/"II";
        only when `plot=True`) and returns (a) the input rows annotated with their
        cluster, and (b) the gene x sample z-score matrix that was clustered.
        """
        target_genes = gly_pathway_genes.copy()
        df_ = df_kidney[df_kidney['gene'].isin(target_genes)].copy()

        stage_col = find_col(df_, 'Path Stage')
        type_col = find_col(df_, 'Type')
        sample_col = find_col(df_, 'Count')
        gene_col = find_col(df_, 'gene')
        value_col = find_col(df_, 'log_tpm')

        if cancer_only:
            df_ = df_[df_[type_col] == 'Cancer'].copy()
            title = title + '-cancer-only'

        df_cancer = df_kidney[df_kidney[type_col] == 'Cancer'].copy()

        for frame in (df_, df_kidney, df_cancer):
            frame['sample number'] = frame[sample_col]
            frame['sample_label'] = frame['sample number'].astype(int).astype(str) + ' (' + frame[type_col].astype(str) + ')'

        columns = (
            df_.sort_values([type_col, stage_col, 'sample number'], ascending=[False, True, True])['sample_label']
            .unique()
        )

        df_2 = df_.pivot_table(index=gene_col, columns='sample_label', values=value_col, aggfunc='mean')
        df_2 = df_2.reindex(columns=columns)

        row_mean = df_2.mean(axis=1)
        row_std = df_2.std(axis=1, ddof=0)
        df_z = df_2.sub(row_mean, axis=0).div(row_std.replace(0, 1), axis=0)
        flat_mask = row_std.eq(0)
        df_z.loc[flat_mask] = df_z.loc[flat_mask]

        link = sch.linkage(df_z.T, method='ward', metric='euclidean')
        col_labels = fcluster(link, t=2, criterion='maxclust')
        cluster_map = {sample: ('I' if cl == 1 else 'II') for sample, cl in zip(df_z.columns, col_labels)}

        cluster_df = pd.DataFrame({
            'sample_label': df_z.columns,
            'cluster_type': [cluster_map[s] for s in df_z.columns],
        })

        df_merge = df_cancer if cancer_only else df_kidney
        df_clust = df_merge.merge(cluster_df, on='sample_label', how='left')

        col_colors = ['red' if cluster_map[s] == 'I' else 'blue' for s in df_z.columns]
        df_z.columns = [f'{col} ({cluster_map[col]})' for col in df_z.columns]

        if plot:
            plt.close('all')
            g = sns.clustermap(
                # Figure3 A: a little taller than square (10 x 11.5).
                df_z, cmap='bwr', figsize=(10, 11.5),
                method='ward', col_cluster=True, col_colors=col_colors, row_cluster=True,
                xticklabels=False, yticklabels=False, cbar_pos=(1, .15, .03, .1),
            )
            plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=20)
            g.ax_heatmap.set_xlabel('')
            g.ax_heatmap.set_ylabel('')
            g.cax.tick_params(labelsize=26)
            g.cax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            g.cax.set_ylabel('z-score', fontsize=26)

            # Columns are always hierarchically clustered here (de novo I/II
            # discovery), so groups can end up interleaved rather than in one
            # contiguous run each -- use a color legend instead of top labels.
            handles = [Patch(facecolor='red', label='Subtype I'), Patch(facecolor='blue', label='Subtype II')]
            g.ax_heatmap.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.90, 1.15),
                                 fontsize=26, frameon=False, handlelength=1.2, handleheight=1.2)

            save_panel(save_dir / f'{title}.PNG', bbox_inches='tight', dpi=600)
            plt.close('all')


        return df_clust, df_z


    def build_ccrcc_clusters(dfs, gly_pathway_genes, save_dir=SAVE_DIR):
        """Run `cancer_glyco_clust` for every ccRCC dataframe variant, as a
        cancer-only view (clustermap = Figure3 A) and a normal+cancer view (no
        plot; its cluster table is used downstream)."""
        ca_all, ca_gly, nca_all, nca_gly = {}, {}, {}, {}
        for name, df_x in dfs:
            print(name)
            df_ca_clu, df_ca_gly_ = cancer_glyco_clust(df_x, gly_pathway_genes, name, cancer_only=True, save_dir=save_dir)
            df_nca_clu, df_nca_gly_ = cancer_glyco_clust(df_x, gly_pathway_genes, name, cancer_only=False, save_dir=save_dir,
                                                          plot=False)
            ca_all[name] = df_ca_clu
            ca_gly[name] = df_ca_gly_
            nca_all[name] = df_nca_clu
            nca_gly[name] = df_nca_gly_
        return ca_all, ca_gly, nca_all, nca_gly


    def overal_survival(df_ca_sub_clin2, title, save_dir=SAVE_DIR):
        """Kaplan-Meier overall survival, cluster I vs. cluster II, with a log-rank test."""
        clin_cols_ori = ['sample_id', 'Dead=1. Alive=0', 'days to death', 'Days to last F/U', 'cluster_type']
        clin_cols = find_cols(clin_cols_ori, df_ca_sub_clin2)

        surv = (
            df_ca_sub_clin2[clin_cols]
            .drop_duplicates('sample_id')
            .set_index('sample_id')
        )
        surv['time'] = np.where(surv[clin_cols[1]] == 1, surv[clin_cols[2]], surv[clin_cols[3]])
        surv = surv.dropna(subset=['time', clin_cols[1], clin_cols[4]])

        grp_I = surv[surv[clin_cols[4]] == 'I']
        grp_II = surv[surv[clin_cols[4]] == 'II']

        if min(len(grp_I), len(grp_II)) < 5:
            raise ValueError("Need at least 5 patients in each cluster for a stable test.")

        lr = logrank_test(
            grp_II['time'], grp_I['time'],
            event_observed_A=grp_II[clin_cols[1]],
            event_observed_B=grp_I[clin_cols[1]],
        )
        print(f"Log-rank p-value = {lr.p_value:.4g}  (chi2 = {lr.test_statistic:.3f})")

        plt.close('all')
        # Figure3 B: square plot area (width = height, excluding tick/axis
        # labels) -- set_box_aspect pins the axes box itself to 1:1.
        fig, ax = plt.subplots(figsize=(11, 11))
        ax.set_box_aspect(1)
        kmf_by_group = {}
        for label, grp, color in [('Subtype I', grp_I, 'red'), ('Subtype II', grp_II, 'blue')]:
            kmf = KaplanMeierFitter()
            kmf.fit(grp['time'], event_observed=grp[clin_cols[1]], label=f"{label} (n={len(grp)})")
            kmf.plot_survival_function(ax=ax, color=color, lw=5, ci_show=False)
            kmf_by_group[label] = kmf

        ax.set_xlabel("Days", fontsize=40)
        ax.set_ylabel("Survival probability", fontsize=40)
        ax.tick_params(axis='x', labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        ax.legend(frameon=False, loc='upper right', fontsize=40)

        # place "p = ..." between the two curves, at the midpoint of their
        # survival probabilities a bit past the median follow-up time
        ylim = ax.get_ylim()
        mid_time = float(surv['time'].quantile(0.65))
        s1 = kmf_by_group['Subtype I'].survival_function_at_times(mid_time).iloc[0]
        s2 = kmf_by_group['Subtype II'].survival_function_at_times(mid_time).iloc[0]
        ax.text(mid_time, (s1 + s2) / 2, f"p = {lr.p_value:.4f}",
                fontsize=40, ha='center', va='center')
        ax.set_ylim(ylim)

        plt.tight_layout()

        outdir = save_dir / 'survival'
        save_panel(outdir / f'{title}_survival.png', bbox_inches='tight', dpi=600)
        plt.close('all')


        return lr


    # -----------------------------------------------------------------------------
    # Pathway-level scoring (which glycosylation pathways differ between groups)
    # -----------------------------------------------------------------------------

    def merge_pathway(df_cancer, df_gly_pathway):
        return df_cancer.merge(df_gly_pathway[['gene', 'pathway class']], on='gene', how='inner')


    def pathway_score(df_ca, col_name, df_gly_pathway):
        """One-way ANOVA per pathway class across the levels of `col_name`.

        For each sample, averages log_tpm within a pathway class, then tests
        whether that per-sample pathway mean differs across the comparison
        groups (e.g. cluster I vs. II, or Normal vs. Cancer).
        """
        df = merge_pathway(df_ca, df_gly_pathway)
        cols = find_cols(['sample', col_name, 'pathway class'], df)

        df_summary = (
            df.groupby(cols, as_index=False)['log_tpm']
              .mean()
              .rename(columns={'log_tpm': 'mean_log_tpm'})
        )
        df_summary = df_summary.sort_values('mean_log_tpm').reset_index(drop=True)

        results = []
        for pw, grp in df_summary.groupby(cols[2]):
            samples_by_cluster = [sub['mean_log_tpm'].values for _, sub in grp.groupby(cols[1])]
            if len(samples_by_cluster) > 1:
                F, p = f_oneway(*samples_by_cluster)
                results.append({'pathway': pw, 'F_stat': F, 'p_value': p, 'nlog10p': -np.log10(p)})

        results_df = pd.DataFrame(results).sort_values('p_value').reset_index(drop=True)
        return df_summary, results_df


    def pathway_sort(df_z, gene=True):
        by = ['pathway class', 'gene'] if gene else ['pathway class']
        return df_z.sort_values(by=by).reset_index(drop=True)


    def pathway_mask(df_ca_score, df_z, k):
        """Keep only rows belonging to the top-k most significant pathway classes."""
        mask = df_ca_score[:k]['pathway']
        return df_z[df_z['pathway class'].isin(mask)].copy()


    def pathway_zscore_matrix(df_ca_score, df_z, k):
        """Gene x sample z-score matrix of the top-k pathways' genes, samples
        ordered Subtype I first, then Subtype II."""
        df_z_top = pathway_mask(df_ca_score, df_z, k)
        df_plot = df_z_top.set_index('gene').drop(columns='pathway class').apply(pd.to_numeric, errors='coerce')
        col_order = sorted(df_plot.columns, key=lambda s: 0 if str(s).strip().endswith('(I)') else 1)
        return df_plot[col_order]


    def plot_gly_pathway(df_ca_score, df_z, k, title, save_dir=SAVE_DIR,
                          height_scale=1.0, fontsize_scale=1.0, cbar_dx=0.0):
        """Z-score heatmap of the top-k pathways' genes, samples split by cluster,
        with pathway-class separators and side labels.

        `height_scale`/`fontsize_scale` scale this panel's height and every text
        element (default 1.0, unchanged). `cbar_dx` shifts the colorbar right
        (figure fraction), away from the pathway labels."""
        df_z_top = pathway_mask(df_ca_score, df_z, k)

        plt.close('all')
        size_k = (10, (15 if k > 12 else 12) * height_scale)
        fig = plt.figure(figsize=size_k)
        heatmap_bottom = 0.20
        heatmap_height = 0.65
        cluster_bar_height = 0.02

        # ax keeps a fixed width so the "Subtype I"/"Subtype II" header (in
        # ax/cluster_ax data coordinates) is never cramped; ax_labels widens
        # with fontsize_scale, and cbar_ax sits right after it with a small
        # fixed gap.
        ax_width = 0.65
        labels_width = 0.2 * fontsize_scale
        labels_x0 = .1 + ax_width + 0.002

        ax = fig.add_axes([.1, heatmap_bottom, ax_width, heatmap_height])
        cluster_ax = fig.add_axes([0.1, heatmap_bottom + heatmap_height + 0.005, ax_width, cluster_bar_height])
        ax_labels = fig.add_axes([labels_x0, heatmap_bottom, labels_width, heatmap_height])
        cbar_ax = fig.add_axes([labels_x0 + labels_width + 0.06 + cbar_dx, heatmap_bottom + 0.02, .02, .2])

        df_plot = df_z_top.set_index('gene').drop(columns='pathway class').apply(pd.to_numeric, errors='coerce')

        col_order = sorted(df_plot.columns, key=lambda s: 0 if str(s).strip().endswith('(I)') else 1)
        cluster_colors = ['red' if s.endswith('(I)') else 'blue' for s in col_order]
        cluster_colors_rgb = [mcolors.to_rgb(c) for c in cluster_colors]

        cluster_ax.imshow([cluster_colors_rgb], aspect='auto')
        cluster_ax.set_xticks([])
        cluster_ax.set_yticks([])

        # "Subtype I" / "Subtype II" header, centered over each cluster's
        # columns, above the cluster color bar at the very top of the figure.
        n_subtype_I = sum(1 for s in col_order if s.endswith('(I)'))
        n_total = len(col_order)
        subtype_centers = [
            ((n_subtype_I - 1) / 2, 'Subtype I'),
            (n_subtype_I + (n_total - n_subtype_I - 1) / 2, 'Subtype II'),
        ]
        subtype_fontsize = 30 * fontsize_scale
        for center, label in subtype_centers:
            cluster_ax.text(center, -0.6, label, ha='center', va='bottom',
                             fontsize=subtype_fontsize, fontweight='bold', clip_on=False,
                             backgroundcolor='none')

        df_plot_clustered = df_plot[col_order]

        sns.heatmap(df_plot_clustered, cmap='bwr', ax=ax, cbar_ax=cbar_ax, cbar=True,
                    xticklabels=False, yticklabels=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        cbar_ax.set_ylabel('z-score', fontsize=26 * fontsize_scale)
        cbar_ax.tick_params(labelsize=26 * fontsize_scale)
        cbar_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        vals = df_z_top['pathway class'].values
        bounds = np.where(vals[:-1] != vals[1:])[0] + 1
        for b in bounds:
            ax.hlines(y=b, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='grey', linewidth=.5, linestyles='--')

        ax_labels.set_ylim(ax.get_ylim())
        ax_labels.set_xlim(-0.02, 1)
        ax_labels.axis("off")

        font_label = (8 if k > 12 else 16) * fontsize_scale
        for group, rows in df_z_top.reset_index().groupby('pathway class'):
            start, end = rows.index[0], rows.index[-1]
            ax_labels.plot([0, 0], [start + 0.25, end + 0.75], color='black', linestyle='-', linewidth=.4)
            mid = 0.5 * (start + end + 1)
            ax_labels.text(0.1, mid, group.replace(', ', ',\n'), ha='left', va='center', fontsize=font_label)

        plt.subplots_adjust(right=0.8)
        save_panel(save_dir / f'{title}_pathway_heatmap.PNG', bbox_inches='tight', dpi=600)
        plt.tight_layout()
        plt.close('all')


        return df_plot_clustered


    # -----------------------------------------------------------------------------
    # PCA (sample-level, on top-pathway genes and per-pathway)
    # -----------------------------------------------------------------------------

    def subtype_(subtype=True):
        return (['I', 'II'], 'cluster_type') if subtype else (['Normal', 'Cancer'], 'Type')


    def plot_pca_pathway(df_ca_score, df_2, k, filename, df_gly_pathway, subtype=True, save_dir=SAVE_DIR,
                          value_scale=None, values=('nlogp', 't', 'fc')):
        """Per-pathway PCA: for each of the top-k pathways, PCA the genes in that
        pathway (samples x genes) and t-test each PC between groups. Plots three
        heatmaps (pathway x PC) of -log10(p), t-score and fold change, annotated
        with FDR-based significance stars.

        `value_scale`: optional {value_name: (height_scale, fontsize_scale)} to
        scale only that one of the three plots (e.g. just 'fc', not 'nlogp'/'t'
        too) -- for when only one of this function's outputs feeds a panel that
        was asked to look different from its siblings.
        """
        df_2 = pathway_mask(df_ca_score, df_2, k).copy()
        name, col_name = subtype_(subtype)

        df_pathway_pca = []
        for pathway, df_pathway in df_gly_pathway.groupby('pathway class'):
            genes = df_pathway['gene'].unique()
            df_pivot = df_2.loc[df_2['gene'].isin(genes)].pivot(index='gene', columns=['sample', col_name], values='log_tpm')
            if df_pivot.shape[0] > 2:
                labels = df_pivot.columns.get_level_values(1) == name[0]
                mypca = PCA(whiten=False, n_components=min(df_pivot.shape[0], 5))
                pca_ = mypca.fit_transform(df_pivot.T)
                t, p = sp.stats.ttest_ind(pca_[~labels], pca_[labels])
                # PC scores are linear combinations of log10(TPM) values, so this
                # difference-of-group-means is itself in log10 units (a PCA-loading
                # -weighted sum of per-gene log10 fold changes); rescale to log2 to
                # match the log2-fold-change convention used elsewhere.
                fc = (pca_[~labels].mean(axis=0) - pca_[labels].mean(axis=0)) * np.log2(10)
                for i in range(pca_.shape[1]):
                    df_pathway_pca.append({'PC': i + 1, 't': t[i], 'p': p[i], 'fc': fc[i],
                                            'nlogp': -np.log10(p[i]), 'pathway': pathway})

        df_pathway_pca = pd.DataFrame(df_pathway_pca)

        _, df_pathway_pca['p_fdr'] = fdrcorrection(df_pathway_pca['p'])
        df_pathway_pca['nlogp_fdr'] = -np.log10(df_pathway_pca['p_fdr'])

        def p_stars(p, ns_label=''):
            if p <= 1e-4:
                return '****'
            elif p <= 1e-3:
                return '***'
            elif p <= 1e-2:
                return '**'
            elif p <= 0.05:
                return '*'
            return ns_label

        df_pathway_pca['sig_stars'] = df_pathway_pca['p_fdr'].apply(p_stars)

        plt.close('all')
        white_red = LinearSegmentedColormap.from_list('white_red', ['white', 'red'])
        stars = df_pathway_pca.pivot(index='pathway', columns='PC', values='sig_stars')
        size_label = (8, 12) if k > 12 else (8, 6)

        outdir = save_dir / 'PCA'

        plots = [
            ('nlogp', white_red, 0, 11, '$-\\log_{10}\\,p$'),
            ('t', 'bwr', -8, 8, '$t$ score'),
            ('fc', 'bwr', -np.log2(10), np.log2(10), '$\\log_2\\,fc$'),
        ]
        plots = [p for p in plots if p[0] in values]  # `values`: which heatmaps to draw
        for val, cmap, vmin, vmax, cbar_label in plots:
            h_scale, f_scale = (value_scale or {}).get(val, (1.0, 1.0))
            plt.figure(figsize=(size_label[0], size_label[1] * h_scale))
            ax_pca = sns.heatmap(
                df_pathway_pca.pivot(index='pathway', columns='PC', values=val),
                cmap=cmap, vmin=vmin, vmax=vmax, annot=stars, fmt='', annot_kws={'size': 20 * f_scale},
                yticklabels=True, cbar=True,
                cbar_kws={'label': cbar_label, 'fraction': 0.025, 'pad': 0.035, 'aspect': 25},
            )
            ax_pca.set_xlabel('Principal component', fontsize=30 * f_scale)
            plt.ylabel('')
            plt.xticks(fontsize=25 * f_scale)
            plt.yticks(fontsize=25 * f_scale)
            cbar_pca = ax_pca.collections[0].colorbar
            cbar_pca.ax.tick_params(labelsize=30 * f_scale)
            cbar_pca.set_label(cbar_label, fontsize=30 * f_scale)
            cbar_pca.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            save_panel(outdir / f'{filename}_{val}_pca.png', bbox_inches='tight', dpi=600)
            plt.close('all')

        return df_pathway_pca


    def df_pivot_gene_in_pathway(pathway, df_2, df_gly_pathway, subtype=True):
        col_name = 'cluster_type' if subtype else 'Type'
        x_gene = df_gly_pathway[df_gly_pathway['pathway class'] == pathway]['gene'].unique()
        df_pivot = df_2[df_2['gene'].isin(x_gene)].drop(columns='pathway class')
        return df_pivot.set_index('gene')


    def top_pathways(df_ca_score, k, exclude_pathways=None):
        """Top-k pathways by significance. `exclude_pathways` drops named
        pathways from the ranking first, so the k slots backfill from lower-
        ranked pathways instead of just being removed after the fact."""
        df = df_ca_score
        if exclude_pathways:
            df = df[~df['pathway'].isin(exclude_pathways)]
        return df[:k]['pathway']


    def PCA_plot_grid_full(df_ca_score, df_ca_clusted, k, df_gly_pathway, filename, subtype=True, save_dir=SAVE_DIR,
                            exclude_pathways=None):
        """Combined figure with all pairwise PC1-4 scatters (6 panels) for each
        of the top-k pathways, one pathway per row,
        instead of separate PNGs per pathway. `exclude_pathways` drops named
        pathways from the ranking before taking the top k (e.g. a pathway with
        too few genes to fill every PC pair, leaving blank panels in its row)
        -- the k slots backfill from the next-ranked pathways instead."""
        import re
        pathways = top_pathways(df_ca_score, k, exclude_pathways=exclude_pathways).tolist()
        name, col_name = subtype_(subtype)
        palette = {name[0]: 'red', name[1]: 'blue'} if subtype else {name[0]: 'black', name[1]: 'red'}

        # (x_pc_index, y_pc_index), 0-based, lower-triangle order
        pc_pairs = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]

        n_rows = len(pathways)
        n_cols = len(pc_pairs)

        plt.close('all')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5.5 * n_rows),
                                  gridspec_kw={'hspace': 0.85, 'wspace': 1.0})
        axes = np.atleast_2d(axes)

        for r, pathway in enumerate(pathways):
            # full pathway name (not truncated): some pathway classes share a
            # prefix before the first comma (e.g. "O-glycan, mucin-type" vs.
            # "O-glycan, not mucin-type"), so trimming there is ambiguous.
            short_name = pathway.strip()
            df_p = df_pivot_gene_in_pathway(pathway, merge_pathway(df_ca_clusted, df_gly_pathway), df_gly_pathway, subtype=True)
            sample_names = df_p.columns
            labels = np.array([name[0] if f"({name[0]})" in str(cn).strip() else name[1] for cn in sample_names])

            ncomp = min(df_p.T.shape[0], df_p.T.shape[1], 4)
            pca_ = None
            if ncomp >= 2:
                pca = PCA(whiten=True, n_components=ncomp)
                pca_ = pca.fit_transform(df_p.T)

            for c, (xi, yi) in enumerate(pc_pairs):
                ax = axes[r, c]
                if pca_ is None or xi >= pca_.shape[1] or yi >= pca_.shape[1]:
                    ax.set_visible(False)
                    continue
                sns.scatterplot(x=pca_[:, xi], y=pca_[:, yi], hue=labels, palette=palette,
                                 ax=ax, s=50, linewidth=0, legend=False)
                ax.set_xlabel(f'PC{xi + 1}', fontsize=28)
                ax.set_ylabel(f'PC{yi + 1}', fontsize=28)
                ax.tick_params(axis='both', labelsize=24)

            # row title, centered above that row's subplots (not per-axes,
            # since a single ax.set_title only centers over one panel)
            left = axes[r, 0].get_position()
            right = axes[r, n_cols - 1].get_position()
            mid_x = (left.x0 + right.x1) / 2
            fig.text(mid_x, left.y1 + 0.005, short_name, ha='center', va='bottom', fontsize=28)

        handles = [
            plt.Line2D([0], [0], marker='o', linestyle='None', color=palette[n], markersize=12)
            for n in name
        ]
        # anchor just below the last row's actual axes bottom (leaving room for
        # its x-tick labels/xlabel), rather than the figure's own bottom edge
        # -- avoids a large blank gap below the last row of subplots.
        last_row_bottom = axes[-1, 0].get_position().y0
        fig.legend(handles, name, loc='upper center', bbox_to_anchor=(0.5, last_row_bottom - 0.02),
                   ncol=len(name), fontsize=26, frameon=True)

        outdir = save_dir / 'PCs'
        save_panel(outdir / f'{filename}.PNG', bbox_inches='tight', dpi=150)
        plt.close('all')


    def _pc1_pc2(df_p, name):
        sample_names = df_p.columns
        labels = np.array([name[0] if f"({name[0]})" in str(c).strip() else name[1] for c in sample_names])
        X = df_p.T
        ncomp = min(X.shape[0], X.shape[1], 2)
        if ncomp < 2:
            return None, None
        pca = PCA(whiten=True, n_components=ncomp)
        pca_ = pca.fit_transform(X)
        return pca_, labels


    def PCA_overview_top_pathways_grid(df_score, df_clusted, df_gly_pathway, subtype, k=5,
                                        overall_title='all pathways', filename='PCs_overview_top_pathways_grid',
                                        save_dir=SAVE_DIR, exclude_pathways=None):
        """One figure for a single comparison (Normal/Cancer or ccRCC Subtype
        I/II): the overall PC1-vs-PC2 scatter (all glycopathway genes
        together), plus a PC1-vs-PC2 panel for each of the top-k pathways --
        `k + 1` panels total, wrapped at 3 columns (2 rows for k=5), one
        shared legend. Used alongside another panel in a shared row (Figure1
        G/J) rather than as its own full-width row -- see
        `PCA_overview_top_pathways_row` for that single-wide-row version
        (Figure1's previous J, Figure3 I).
        """
        name, _ = subtype_(subtype)
        palette = {name[0]: 'red', name[1]: 'blue'} if subtype else {name[0]: 'black', name[1]: 'red'}
        group_label = [f'Subtype {n}' for n in name] if subtype else list(name)

        panels = []

        pca_overall, labels_overall = _pc1_pc2(df_clusted, name)
        panels.append((overall_title, pca_overall, labels_overall))

        for pathway in top_pathways(df_score, k, exclude_pathways=exclude_pathways).tolist():
            df_p = df_pivot_gene_in_pathway(pathway, merge_pathway(df_clusted, df_gly_pathway),
                                             df_gly_pathway, subtype=subtype)
            pca_, labels = _pc1_pc2(df_p, name)
            # full pathway name (not truncated): some pathway classes share a
            # prefix before the first comma (e.g. "O-glycan, mucin-type" vs.
            # "O-glycan, not mucin-type"), so trimming there is ambiguous.
            panels.append((pathway.strip(), pca_, labels))

        ncols = 3
        nrows = int(np.ceil(len(panels) / ncols))

        plt.close('all')
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                                  gridspec_kw={'hspace': 1.0, 'wspace': 0.45})
        axes = np.atleast_2d(axes)

        for idx, (title, pca_, labels) in enumerate(panels):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            if pca_ is None:
                ax.set_visible(False)
                continue
            sns.scatterplot(x=pca_[:, 0], y=pca_[:, 1], hue=labels, palette=palette,
                             hue_order=name, ax=ax, s=60, linewidth=0, legend=False)
            # wrap onto two lines if long enough to overflow the panel width
            if ',' in title and len(title) > 18:
                title = title.replace(', ', ',\n', 1)
            ax.set_title(title, fontsize=30, pad=14)
            ax.set_xlabel('PC1', fontsize=30)
            ax.set_ylabel('PC2', fontsize=30, labelpad=2)
            ax.tick_params(axis='both', labelsize=30)
            ax.set_box_aspect(1)

        for idx in range(len(panels), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        fig.tight_layout()

        # legend bottom-aligned with the bottom row's plot (axes) area, not as
        # low as the x-axis label text below it
        last_row = nrows - 1
        last_col = min(len(panels) - 1, ncols - 1) if last_row == nrows - 1 else ncols - 1
        bottom_ax = axes[last_row, last_col] if axes[last_row, last_col].get_visible() else axes[last_row, 0]
        bottom_y = bottom_ax.get_position().y0

        handles = [plt.Line2D([0], [0], marker='o', linestyle='None', color=palette[n], markersize=9) for n in name]
        # fontsize matches the per-panel subtitles (ax.set_title(..., fontsize=30) above)
        fig.legend(handles, group_label, loc='upper center', bbox_to_anchor=(0.5, bottom_y - 0.14),
                   ncol=len(name), fontsize=30, frameon=False)

        outdir = save_dir / 'PCs'
        save_panel(outdir / f'{filename}.png', bbox_inches='tight', dpi=300)
        plt.close('all')


    def PCA_overview_top_pathways_row(df_score, df_clusted, df_gly_pathway, subtype, k=5,
                                       overall_title='all pathways', filename='PCs_overview_top_pathways_row',
                                       save_dir=SAVE_DIR, exclude_pathways=None):
        """One figure for a single comparison (Normal/Cancer or ccRCC Subtype
        I/II): the overall PC1-vs-PC2 scatter (all glycopathway genes
        together -- the same view as `PCs_ccRCC_norm_cancer.png` /
        `PCs_ccRCC_subtypes.png`), plus a PC1-vs-PC2 panel for each of the
        top-k pathways -- `k + 1` panels total laid out in a single wide row,
        one shared legend. Used as its own dedicated full-width row in a
        manuscript figure (Figure1 J, Figure3 I).
        """
        name, _ = subtype_(subtype)
        palette = {name[0]: 'red', name[1]: 'blue'} if subtype else {name[0]: 'black', name[1]: 'red'}
        group_label = [f'Subtype {n}' for n in name] if subtype else list(name)

        panels = []

        pca_overall, labels_overall = _pc1_pc2(df_clusted, name)
        panels.append((overall_title, pca_overall, labels_overall))

        for pathway in top_pathways(df_score, k, exclude_pathways=exclude_pathways).tolist():
            df_p = df_pivot_gene_in_pathway(pathway, merge_pathway(df_clusted, df_gly_pathway),
                                             df_gly_pathway, subtype=subtype)
            pca_, labels = _pc1_pc2(df_p, name)
            panels.append((pathway.strip(), pca_, labels))

        ncols = len(panels)
        nrows = 1

        plt.close('all')
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                                  gridspec_kw={'hspace': 1.0, 'wspace': 0.45})
        axes = np.atleast_2d(axes)

        for idx, (title, pca_, labels) in enumerate(panels):
            ax = axes[0, idx]
            if pca_ is None:
                ax.set_visible(False)
                continue
            sns.scatterplot(x=pca_[:, 0], y=pca_[:, 1], hue=labels, palette=palette,
                             hue_order=name, ax=ax, s=60, linewidth=0, legend=False)
            if ',' in title and len(title) > 18:
                title = title.replace(', ', ',\n', 1)
            ax.set_title(title, fontsize=30, pad=14)
            ax.set_xlabel('PC1', fontsize=30)
            ax.set_ylabel('PC2', fontsize=30, labelpad=2)
            ax.tick_params(axis='both', labelsize=30)
            # square plot box (width == height), not just a square nominal
            # cell -- the title/tick/label text around each axes would
            # otherwise eat into the height more than the width, leaving a
            # visibly taller-than-wide box despite the equal 4.5in x 4.5in
            # cell allocation above.
            ax.set_box_aspect(1)

        fig.tight_layout()

        bottom_ax = axes[0, ncols - 1] if axes[0, ncols - 1].get_visible() else axes[0, 0]
        bottom_y = bottom_ax.get_position().y0

        handles = [plt.Line2D([0], [0], marker='o', linestyle='None', color=palette[n], markersize=9) for n in name]
        fig.legend(handles, group_label, loc='upper center', bbox_to_anchor=(0.5, bottom_y - 0.14),
                   ncol=len(name), fontsize=30, frameon=False)

        outdir = save_dir / 'PCs'
        save_panel(outdir / f'{filename}.png', bbox_inches='tight', dpi=300)
        plt.close('all')


    def _plot_gene_zscore_heatmap(df_p, filename, subtype=True, save_dir=SAVE_DIR):
        """Shared plotting body for a gene x sample z-score heatmap, samples
        split into the two comparison groups with a divider and top labels."""
        name, _ = subtype_(subtype)
        group_label = [f'Subtype {n}' for n in name] if subtype else list(name)
        col_order = sorted(df_p.columns, key=lambda s: 0 if f"({name[0]})" in str(s).strip() else 1)
        df_p = df_p[col_order]
        n0 = sum(1 for c in col_order if f"({name[0]})" in str(c).strip())

        y_font = 26

        plt.close('all')
        fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(col_order) + 4), max(6, 0.4 * len(df_p.index) + 2)))
        vmax = float(np.nanmax(np.abs(df_p.to_numpy(dtype=float))))
        sns.heatmap(df_p, cmap='bwr', center=0, vmin=-vmax, vmax=vmax, ax=ax, cbar=False,
                    linewidths=0.3, linecolor='lightgrey', xticklabels=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.setp(ax.get_yticklabels(), fontsize=y_font, rotation=0)
        ax.axvline(n0, color='black', lw=2, linestyle='--')

        for center, label in [(n0 / 2, group_label[0]), (n0 + (len(col_order) - n0) / 2, group_label[1])]:
            ax.text(center, -0.5, label, ha='center', va='bottom', fontsize=44, fontweight='bold', clip_on=False)

        fig.tight_layout()
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.02, pos.height * 0.3])
        cbar = fig.colorbar(ax.collections[0], cax=cbar_ax, label='z-score')
        cbar.ax.tick_params(labelsize=30)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        cbar.set_label('z-score', fontsize=30)

        outdir = save_dir / 'PCs'
        save_panel(outdir / f'{filename}.PNG', bbox_inches='tight', pad_inches=0.03, dpi=300)
        plt.close('all')

        return df_p


    def sialyltransferase_genes(df_gly_pathway):
        """Sialyltransferase gene family (ST3GAL*, ST6GAL*, ST6GALNAC*, ST8SIA*),
        which spans several pathway classes rather than being its own class."""
        mask = df_gly_pathway['gene'].astype(str).str.match(r'^ST(3GAL|6GAL|6GALNAC|8SIA)\d*$', na=False)
        return df_gly_pathway.loc[mask, 'gene'].unique().tolist()


    def plot_gene_list_zscore_heatmap(gene_list, filename, df_plot_clustered, df_gly_pathway, subtype=True, save_dir=SAVE_DIR):
        """Z-score heatmap of an explicit gene list (e.g. a gene family that
        spans several pathway classes). Rows follow `gene_list`'s given order (e.g.
        sorted by significance) rather than table insertion order."""
        merged = merge_pathway(df_plot_clustered, df_gly_pathway)
        df_p = merged[merged['gene'].isin(gene_list)].drop(columns='pathway class').set_index('gene').reindex(gene_list)
        return _plot_gene_zscore_heatmap(df_p, filename, subtype=subtype, save_dir=save_dir)


    # -----------------------------------------------------------------------------
    # Per-gene scoring (mean-level t-test, and per-sample z-scores for CyberSortX)
    # -----------------------------------------------------------------------------

    def gene_score(df_2, df_gly_pathway, subtype=True):
        """Welch t-test per gene between the two comparison groups (mean-level)."""
        rows = []
        for gene, df_gene in df_gly_pathway.groupby('gene'):
            pathway = df_gene['pathway class'].iloc[0]
            if subtype:
                name, col_name = ["I", "II"], 'cluster_type'
            else:
                name, col_name = ["Normal", "Cancer"], 'Type'

            dfA = df_2.loc[(df_2['gene'] == gene) & (df_2[col_name] == name[0]), 'log_tpm']
            dfB = df_2.loc[(df_2['gene'] == gene) & (df_2[col_name] == name[1]), 'log_tpm']

            t, p = stats.ttest_ind(dfB, dfA, equal_var=False, nan_policy='omit')

            # dfA/dfB are log10(TPM), so this mean-difference is itself in
            # log10 units; rescale by log2(10) to report log2 fold change,
            # matching the convention used elsewhere (e.g. the volcano plots).
            log2_fc = (dfB.mean() - dfA.mean()) * np.log2(10)

            rows.append({
                'gene': gene, 'pathway': pathway,
                'log_fc': log2_fc,
                'fc': 2 ** log2_fc,
                't': t, 'p': p, 'nlogp': -np.log10(p),
                'direction': 'increasing' if t > 0 else 'decreasing',
            })

        df_pathway_scores = pd.DataFrame(rows).dropna().sort_values('p', ascending=True)
        return df_pathway_scores


    def gene_score_per_sample(df_2, df_gly_pathway, value_col='log_tpm', sample_col='Abbreviation_x', cancer=True):
        """Per-sample, per-gene z-score (z-scored across all samples for that gene).

        This long-format table is passed on to stage_04.
        """
        rows = []
        pathway_map = df_gly_pathway[['gene', 'pathway class']].drop_duplicates()
        df_use = df_2.merge(pathway_map, on='gene', how='inner').copy()
        type_cols = rw_function.col_type(cancer)

        for gene, df_gene in df_use.groupby('gene'):
            vals = df_gene[value_col]
            mu = vals.mean()
            sigma = vals.std(ddof=0)

            if pd.isna(sigma) or sigma == 0:
                z = pd.Series(np.nan, index=df_gene.index)
            else:
                z = (vals - mu) / sigma

            tmp = df_gene[['gene', 'pathway class', sample_col, type_cols]].copy()
            tmp['zscore'] = z.values
            tmp = tmp.rename(columns={sample_col: 'sample'})
            rows.append(tmp)

        df_pathway_scores = pd.concat(rows, ignore_index=True)
        return df_pathway_scores.dropna(subset=['zscore'])


    def plot_gene_score(df_ca_score, df_pathway_scores, k, filename, save_dir=SAVE_DIR,
                         height_scale=1.0, fontsize_scale=1.0, legend_dx=0.0, legend_dy=0.0,
                         width_scale=1.0):
        """Dot plot of per-gene log2(fold change) and -log10(p), split by direction,
        for genes in the top-k pathways. `height_scale`/`fontsize_scale` scale
        this panel's height and every text element (default 1.0, unchanged).
        `legend_dx`/`legend_dy` nudge the top legend (figure fraction; negative
        = left/down) -- dy<0 pulls it closer to the plot. `width_scale` widens
        the fc/p subplots (figure width; default 1.0, unchanged)."""
        df = pathway_mask(df_ca_score, df_pathway_scores, k)

        plt.close('all')
        size_label = (6 * width_scale, (8 if k > 12 else 3) * height_scale)
        fig, ax = plt.subplots(1, 2, figsize=size_label, sharey=True)

        sns.scatterplot(data=df, x='log_fc', y='pathway', hue='direction',
                         palette={'increasing': 'red', 'decreasing': 'blue'}, legend=False, s=20, ax=ax[0])
        ax[0].axvline(0, c='0.1', lw=2, ls='--', zorder=0)
        ax[0].set_xlabel('$\\log_2$ fc', fontsize=20 * fontsize_scale)

        sns.scatterplot(data=df, x='nlogp', y='pathway', hue='direction',
                         palette={'increasing': 'red', 'decreasing': 'blue'},
                         hue_order=['increasing', 'decreasing'], s=20, ax=ax[1])

        cutoff = .01
        ax[1].axvline(-np.log10(cutoff), c='0.1', lw=2, ls='--', zorder=0)
        ax[0].set_ylabel('')

        # On top, horizontally centered over both subplots combined (not just
        # ax[1]) and on one line -- anchored above the axes' own top edge, so
        # it can't overlap any plotted point regardless of where each
        # pathway's dots happen to fall.
        handles, labels = ax[1].get_legend_handles_labels()
        ax[1].get_legend().remove()
        pos0, pos1 = ax[0].get_position(), ax[1].get_position()
        center_x = (pos0.x0 + pos1.x1) / 2
        top_y = max(pos0.y1, pos1.y1)
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(center_x + legend_dx, top_y + legend_dy),
                   ncol=len(labels), frameon=False, markerscale=1.5, handletextpad=0.3,
                   fontsize=22 * fontsize_scale)

        for ax_ in ax:
            ax_.tick_params(axis='both', labelsize=18 * fontsize_scale)
        ax[1].set_xlabel('$-\\log_{10}\\,p$', fontsize=20 * fontsize_scale)

        for spine in ['top', 'right']:
            for ax_ in ax:
                ax_.spines[spine].set_visible(False)

        outdir = save_dir / 'gscore'
        save_panel(outdir / f'{filename}_gene_score.png', bbox_inches='tight', dpi=300)
        plt.close('all')


    def top_genes_across_pathways_heatmap(df_score, df_gene_score, df_gene_score_psample, k_pathways,
                                           genes_per_pathway, filename, subtype=True, save_dir=SAVE_DIR,
                                           cbar_offset=0.25, exclude_genes=None, exclude_pathways=None,
                                           height_scale=1.0, fontsize_scale=1.0, cbar_fontsize_scale=1.0):
        """One combined z-score heatmap of the top-`genes_per_pathway` most
        significant genes (by Welch t-test p-value, from `gene_score`) within
        each of the top-`k_pathways` pathways (by ANOVA p-value, from
        `pathway_score`/`top_pathways`).

        Genes are grouped into contiguous row blocks by pathway (in top-k
        pathway order), gene names labeled on the left (the heatmap's normal
        y-axis), pathway name labeled on the right of its block, with samples
        split into the two comparison groups (top label + divider).

        `height_scale`/`fontsize_scale` scale this panel's height and every
        text element (default 1.0, unchanged); `cbar_fontsize_scale` further
        scales just the colorbar's tick labels and "z-score" label.
        """
        name, col_name = subtype_(subtype)
        group_label = [f'Subtype {n}' for n in name] if subtype else list(name)

        pathways = top_pathways(df_score, k_pathways, exclude_pathways=exclude_pathways).tolist()

        top_genes = []
        seen = set(exclude_genes) if exclude_genes else set()
        pathway_blocks = []
        for pathway in pathways:
            sub = df_gene_score[df_gene_score['pathway'] == pathway].sort_values('p', ascending=True)
            genes = [g for g in sub['gene'].tolist() if g not in seen][:genes_per_pathway]
            seen.update(genes)
            top_genes.extend(genes)
            pathway_blocks.append((pathway, len(genes)))

        df_top = df_gene_score_psample[df_gene_score_psample['gene'].isin(top_genes)].copy()
        mat = df_top.pivot_table(index='gene', columns='sample', values='zscore', aggfunc='mean').reindex(top_genes)

        group_map = df_top.drop_duplicates('sample').set_index('sample')[col_name]
        order = group_map.map({n: i for i, n in enumerate(name)}).sort_values().index
        order = [c for c in order if c in mat.columns]
        mat = mat[order]
        n0 = sum(1 for c in order if group_map[c] == name[0])

        y_font = 30 * fontsize_scale

        plt.close('all')
        fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(order) + 6),
                                         max(8, 0.46 * len(mat.index) + 2) * height_scale))
        vmax = float(np.nanmax(np.abs(mat.to_numpy(dtype=float))))
        sns.heatmap(mat, cmap='bwr', center=0, vmin=-vmax, vmax=vmax, ax=ax, cbar=False,
                    linewidths=0.3, linecolor='lightgrey', xticklabels=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.setp(ax.get_yticklabels(), fontsize=y_font, rotation=0)
        ax.axvline(n0, color='black', lw=2, linestyle='--')

        for center, label in [(n0 / 2, group_label[0]), (n0 + (len(order) - n0) / 2, group_label[1])]:
            ax.text(center, -0.52, label, ha='center', va='bottom', fontsize=44 * fontsize_scale, fontweight='bold', clip_on=False)

        # pathway name to the right of its row block, with a divider between blocks
        n_cols = len(order)
        row_start = 0
        for pathway, size in pathway_blocks:
            row_end = row_start + size
            if row_start > 0:
                ax.axhline(row_start, color='grey', lw=2.5, linestyle='--')
            center = (row_start + row_end) / 2
            # wrap onto two lines if long enough to be awkward on one line --
            # split on the first comma when there is one (e.g. "glycolipid,
            # lacto/neolacto series"), otherwise on the first space (e.g.
            # "terminal modification", which has no comma to split on).
            if ',' in pathway and len(pathway) > 18:
                label = pathway.replace(', ', ',\n', 1)
            elif ' ' in pathway and len(pathway) > 18:
                label = pathway.replace(' ', '\n', 1)
            else:
                label = pathway
            ax.text(n_cols + 0.3, center, label, ha='left', va='center', fontsize=31 * fontsize_scale, clip_on=False)
            row_start = row_end

        fig.tight_layout()
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1 + cbar_offset, pos.y0, 0.015, pos.height * 0.3])
        cbar = fig.colorbar(ax.collections[0], cax=cbar_ax, label='z-score')
        cbar_font = 33 * fontsize_scale * cbar_fontsize_scale
        cbar.ax.tick_params(labelsize=cbar_font)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        cbar.set_label('z-score', fontsize=cbar_font)

        outdir = save_dir / 'gscore'
        save_panel(outdir / f'{filename}.PNG', bbox_inches='tight', pad_inches=0.03, dpi=300)
        plt.close('all')

        return mat


    # -----------------------------------------------------------------------------
    # CIBERSORTx immune-fraction integration
    # -----------------------------------------------------------------------------

    def df_immune_merge(filename, df_ccrcc_mito_all, cancer=True, data_dir=DATA_DIR):
        """Load a CIBERSORTx fraction table and attach the cluster/Type label."""
        col_ = 'patient_id'
        df_ca_immune = pd.read_csv(data_dir / f"{filename}.csv")
        df_ca_immune[col_] = df_ca_immune['Mixture'].str[:-1]
        df_ca_immune['Type'] = 'Cancer' if cancer else 'Normal'
        df_ = df_ca_immune.drop(columns=['RMSE', 'P-value', 'Correlation'])

        col_names = ['patient_id', 'Type', 'Count_x']
        df_clust_type = df_ccrcc_mito_all[col_names].drop_duplicates()

        return pd.merge(df_, df_clust_type, on=['patient_id', 'Type'], how='inner')


    CLUSTER_TABLE_COLUMNS = [
        'gene', 'sample', 'sample_id', 'log_tpm', 'Type', 'cluster_type',
        'Count', 'Path Stage', 'Dead=1. Alive=0', 'days to death', 'Days to last F/U',
    ]


    def trim_cluster_table(df):
        """De-duplicate and slim down a cluster-annotated long table before saving.

        `df_cancer_type` merges the sample table with `df_clin` and then again
        with `df_ccrcc`, so every clinical column ends up tripled on disk as
        `col`, `col_x`, `col_y` (identical values, just repeated) -- multiplied
        by one row per gene per sample, this is what blows these tables up to
        hundreds of MB. Keep only the columns later stages read
        (`CLUSTER_TABLE_COLUMNS`), each resolved to its first match via
        `find_col` (unsuffixed, then `_x`, then `_y`) -- the same lookup those
        stages use, so this only reduces size.
        """
        out = {}
        for base_name in CLUSTER_TABLE_COLUMNS:
            col = find_col(df, base_name)
            if col is not None:
                out[base_name] = df[col]
        return pd.DataFrame(out)


    def plot_mean_immune(df_ca_immune_type, filename, cancer=True, save_dir=SAVE_DIR):
        """Heatmap of mean z-scored immune-cell fraction per group (cluster or Type)."""
        immune_cols = df_ca_immune_type.columns[1:-4].tolist()
        df_ca_immune_type = df_ca_immune_type.copy()
        df_ca_immune_type[immune_cols] = df_ca_immune_type[immune_cols].astype(np.float32)

        X = df_ca_immune_type[immune_cols].to_numpy(dtype=np.float32).copy()
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, ddof=0, keepdims=True)
        sigma[sigma == 0] = 1e-9
        df_ca_immune_type[immune_cols] = (X - mu) / sigma

        col_ = find_col_name(cancer)
        df_im_gly_clu = df_ca_immune_type.groupby(col_, observed=True)[immune_cols].mean().T

        if not cancer:
            df_im_gly_clu = df_im_gly_clu.reindex(columns=['Normal', 'Cancer'])
        else:
            df_im_gly_clu = df_im_gly_clu.rename(columns={'I': 'Subtype I', 'II': 'Subtype II'})

        x_font = 20

        plt.close('all')
        fig, ax = plt.subplots(figsize=(3.75, 9))
        sns.heatmap(df_im_gly_clu, cmap='bwr', annot=True, fmt='.2f',
                    annot_kws={'fontsize': 14}, ax=ax, cbar=False)
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=x_font)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticklabels(df_im_gly_clu.index, rotation=0, fontsize=x_font)

        fig.tight_layout()
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.03, pos.height * 0.3])
        cbar = fig.colorbar(ax.collections[0], cax=cbar_ax)
        cbar.set_label('z-score', fontsize=26)
        cbar.ax.tick_params(labelsize=26)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        outdir = save_dir / 'immune'
        save_panel(outdir / f'{filename}_immune_cell_fraction_mean.PNG', bbox_inches='tight', dpi=300)
        plt.close('all')


    def plot_ind_immune(df_slice, filename, cancer=True, save_dir=SAVE_DIR):
        """Per-sample immune-cell z-score heatmap, samples sorted by group."""
        col_ = find_col_name(cancer)
        immune_cols = df_slice.columns[1:-4].tolist()
        order = np.argsort(df_slice[col_].to_numpy())
        X = df_slice[immune_cols].to_numpy(dtype=np.float32)[order]
        cluster_vec = df_slice[col_].to_numpy()[order]

        split_pos = np.argmax(cluster_vec != cluster_vec[0])

        fig_label = (10, 9) if cancer else (10, 12)
        plt.close('all')
        fig, ax = plt.subplots(figsize=fig_label)
        im = ax.imshow(X, aspect='auto', cmap='bwr', interpolation='none')

        ax.set_xticks(np.arange(len(immune_cols)))
        ax.set_xticklabels(immune_cols, rotation=90, fontsize=20)
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax.axhline(split_pos - 0.5, color='black', linestyle='--', lw=2)

        label_ = ['Subtype I', 'Subtype II'] if cancer else ['Cancer', 'Normal']
        ax.text(-0.5, split_pos / 2, label_[0], va='center', ha='right', fontsize=40, weight='bold', rotation=90)
        ax.text(-0.5, split_pos + (len(order) - split_pos) / 2, label_[1],
                va='center', ha='right', fontsize=40, weight='bold', rotation=90)

        fig.tight_layout()

        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.02, pos.height * 0.3])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('z-score', fontsize=26)
        cbar.ax.tick_params(labelsize=26)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        outdir = save_dir / 'immune'
        save_panel(outdir / f'{filename}_immune_fraction_individual.PNG', bbox_inches='tight', dpi=300)
        plt.close('all')


    # -----------------------------------------------------------------------------
    # Gene-of-interest and custom gene-list expression heatmaps
    # -----------------------------------------------------------------------------

    def gene_interest_heatmap(df_cancer_subtype, filename, gene_name, cancer=True, save_dir=SAVE_DIR,
                              save_mean=True):
        """Per-sample z-score heatmap + cluster-mean heatmap for genes whose name
        contains `gene_name` (e.g. 'siglec')."""
        plt.close('all')
        col_ = find_col_name(cancer)
        col_abbrev = 'Abbreviation' if cancer else 'Abbreviation_x'

        mask = df_cancer_subtype['gene'].str.contains(gene_name, case=False, na=False)
        df_siglec = df_cancer_subtype[mask].copy()
        if df_siglec.empty:
            raise ValueError(f"No genes containing {gene_name} found!")

        mat = df_siglec.pivot_table(index='gene', columns=col_abbrev, values='log_tpm', aggfunc='mean')

        clu_idx = df_siglec[[col_abbrev, col_]].drop_duplicates(subset=col_abbrev).set_index(col_abbrev)[col_]
        my_col_order = desired_order(cancer)
        order = clu_idx.map({name: i for i, name in enumerate(my_col_order)}).sort_values().index
        order = [x for x in order if x in mat.columns]
        mat = mat[order]

        # genes ordered by significance (Welch t-test between the two groups),
        # most significant at the top.
        group0_cols = [c for c in order if clu_idx[c] == my_col_order[0]]
        group1_cols = [c for c in order if clu_idx[c] == my_col_order[1]]
        _, gene_pvals = ttest_ind(mat[group0_cols], mat[group1_cols], axis=1, equal_var=False, nan_policy='omit')
        mat = mat.loc[pd.Series(gene_pvals, index=mat.index).sort_values().index]

        z = (mat.subtract(mat.mean(axis=1), axis=0)
                .div(mat.std(axis=1, ddof=0), axis=0)
                .fillna(0)
                .to_numpy(dtype=np.float32))

        vmax = np.nanmax(np.abs(z))
        f_size = (10, 6) if cancer else (18, 6)
        fig, ax = plt.subplots(figsize=f_size)
        im = ax.imshow(z, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax, interpolation='none')

        ax.set_xticks([])
        y_font = 16
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=y_font)

        split = np.argmax(clu_idx.loc[order].ne(clu_idx.loc[order].iloc[0]).to_numpy())
        ax.axvline(split - 0.5, color='black', lw=2, linestyle='--')

        # top group-header labels (Normal/Cancer or Subtype I/II) + color line
        # at the very top, one per contiguous run of same-group columns.
        if cancer:
            group_of = [str(clu_idx[s]).strip() for s in order]
            color_of = {'I': 'red', 'II': 'blue'}
            display_map = {'I': 'Subtype I', 'II': 'Subtype II'}
        else:
            group_of = ['Normal' if str(s).strip().endswith('B') else 'Cancer' for s in order]
            color_of = {'Normal': 'black', 'Cancer': 'red'}
            display_map = {}

        n = len(group_of)
        seg_start = 0
        for i in range(1, n + 1):
            if i == n or group_of[i] != group_of[seg_start]:
                g = group_of[seg_start]
                label = display_map.get(g, g)
                ax.plot([seg_start - 0.5, i - 0.5], [-1.0, -1.0],
                        color=color_of[g], linewidth=4, solid_capstyle='butt', clip_on=False)
                center = (seg_start + i - 1) / 2
                # Smaller than the matching label elsewhere (e.g. genelist_heatmap's
                # 40pt) because this panel gets stretched more in the composite
                # manuscript figure (wider assigned column for its aspect ratio),
                # which would otherwise make its baked-in text render larger on the page.
                ax.text(center, -1.3, label, ha='center', va='bottom',
                        fontsize=20, fontweight='bold', clip_on=False)
                seg_start = i

        fig.tight_layout()

        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.02, pos.height * 0.3])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('z-score', fontsize=26)
        cbar.ax.tick_params(labelsize=26)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        outdir = save_dir / 'immune'
        save_panel(outdir / f'{filename}_{gene_name}_individual.PNG', dpi=600, bbox_inches='tight')
        plt.close('all')
        if not save_mean:  # per-group mean heatmap not needed for this comparison
            return

        cluster_mean = (
            df_siglec.groupby(['gene', col_])['log_tpm']
            .mean()
            .unstack(col_)
            .reindex(columns=my_col_order)
            .loc[mat.index]
        )
        if cancer:
            cluster_mean = cluster_mean.rename(columns={'I': 'Subtype I', 'II': 'Subtype II'})

        fig2, ax2 = plt.subplots(figsize=(4, 8))
        sns.heatmap(cluster_mean, cmap='bwr', annot=True, fmt='.2f',
                    annot_kws={'fontsize': 16}, ax=ax2, cbar=False)
        plt.setp(ax2.get_xticklabels(), rotation=90, ha='center', fontsize=40)
        ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=20, rotation=0)
        ax2.set_ylabel("")
        ax2.set_xlabel("")

        fig2.tight_layout()
        pos2 = ax2.get_position()
        cbar_ax2 = fig2.add_axes([pos2.x1 + 0.02, pos2.y0, 0.04, pos2.height * 0.3])
        cbar2 = fig2.colorbar(ax2.collections[0], cax=cbar_ax2)
        cbar2.set_label('log(TPM)', fontsize=26)
        cbar2.ax.tick_params(labelsize=26)
        cbar2.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        save_panel(outdir / f'{filename}_{gene_name}_mean.PNG', dpi=600, bbox_inches='tight')
        plt.close('all')


    def genelist_heatmap(
        df_cancer_subtype, title, filename, genes=None, cancer=True, k=None,
        sort_by_p=True, pval_colname='p_value', use_abs_effect=False, save_dir=SAVE_DIR,
    ):
        """Z-score heatmap for a custom gene list (e.g. immune-regulated genes),
        genes ordered by significance (Welch t-test, group A vs. group B), with a
        top color bar marking each sample's group."""
        plt.close("all")

        col_ = find_col_name(cancer)
        col_abbrev = 'Abbreviation' if cancer else 'Abbreviation_x'
        my_col_order = desired_order(cancer)
        group_label = "cluster type" if "cluster" in str(col_).lower() else "type"

        df_ = df_cancer_subtype.copy()
        df_['gene'] = df_['gene'].astype(str)

        if genes is not None:
            genes = [str(g) for g in genes]
            df_ = df_[df_['gene'].isin(genes)].copy()
            if df_.empty:
                raise ValueError("No genes from the provided gene list were found.")

        df_ = df_.dropna(subset=['gene', 'log_tpm', col_abbrev, col_]).copy()
        if df_.empty:
            raise ValueError("No valid rows remain after filtering.")

        df_ = df_[df_[col_].isin(my_col_order)].copy()
        if df_.empty:
            raise ValueError(f"No rows found for comparison groups: {my_col_order}")

        def gene_stats(g):
            a = g.loc[g[col_] == my_col_order[0], 'log_tpm'].dropna().to_numpy()
            b = g.loc[g[col_] == my_col_order[1], 'log_tpm'].dropna().to_numpy()
            if len(a) < 2 or len(b) < 2:
                pval, effect = np.nan, np.nan
            else:
                _, pval = ttest_ind(a, b, equal_var=False, nan_policy='omit')
                effect = np.mean(b) - np.mean(a)
            return pd.Series({
                pval_colname: pval, 'effect': effect,
                f'mean_{my_col_order[0]}': np.mean(a) if len(a) else np.nan,
                f'mean_{my_col_order[1]}': np.mean(b) if len(b) else np.nan,
            })

        stats_df = df_.groupby('gene', sort=False).apply(gene_stats).reset_index()

        if genes is not None and not sort_by_p:
            gene_order = [g for g in genes if g in stats_df['gene'].values]
        else:
            stats_df = stats_df.copy()
            stats_df['_p_sort'] = stats_df[pval_colname].fillna(np.inf)
            if use_abs_effect:
                stats_df['_effect_sort'] = stats_df['effect'].abs().fillna(-np.inf)
                stats_df = stats_df.sort_values(by=['_p_sort', '_effect_sort', 'gene'], ascending=[True, False, True])
            else:
                stats_df = stats_df.sort_values(by=['_p_sort', 'gene'], ascending=[True, True])
            gene_order = stats_df['gene'].tolist()

        if k is not None:
            gene_order = gene_order[:k]
        if len(gene_order) == 0:
            raise ValueError("No genes available to plot after filtering/sorting.")

        df_gene = df_[df_['gene'].isin(gene_order)].copy()

        mat = (
            df_gene.pivot_table(index='gene', columns=col_abbrev, values='log_tpm', aggfunc='mean')
            .reindex(gene_order)
        )

        clu_idx = df_gene[[col_abbrev, col_]].drop_duplicates(subset=col_abbrev).set_index(col_abbrev)[col_]
        order = clu_idx.map({name: i for i, name in enumerate(my_col_order)}).sort_values().index.tolist()
        order = [x for x in order if x in mat.columns]
        mat = mat.loc[:, order]

        row_mean = mat.mean(axis=1)
        row_std = mat.std(axis=1, ddof=0).replace(0, np.nan)
        z = mat.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0)

        vmax = np.nanmax(np.abs(z.to_numpy()))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1

        if cancer:
            group_colors = {my_col_order[0]: 'red', my_col_order[1]: 'blue'}
        else:
            group_colors = {my_col_order[0]: 'black', my_col_order[1]: 'red'}

        group_labels = clu_idx.loc[order]
        top_bar_vals = np.array([my_col_order.index(x) for x in group_labels]).reshape(1, -1)
        top_bar_cmap = ListedColormap([group_colors[g] for g in my_col_order])

        fig = plt.figure(figsize=(12, max(8, 0.45 * len(gene_order) + 4)))

        ax_top = fig.add_axes([0.10, 0.9, 0.7, 0.01])
        ax_top.imshow(top_bar_vals, aspect='auto', cmap=top_bar_cmap, interpolation='none')
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for spine in ax_top.spines.values():
            spine.set_visible(False)

        print(group_label)
        if len(order) > 0:
            boundaries = []
            start = 0
            arr = group_labels.to_numpy()
            for i in range(1, len(arr)):
                if arr[i] != arr[i - 1]:
                    boundaries.append((start, i - 1, arr[i - 1]))
                    start = i
            boundaries.append((start, len(arr) - 1, arr[-1]))

            subtype_label_display = {'I': 'Subtype I', 'II': 'Subtype II'} if cancer else {}
            for s, e, label in boundaries:
                center = (s + e) / 2
                display_label = subtype_label_display.get(str(label), str(label))
                ax_top.text(center, -6, display_label, ha='center', va='top', fontsize=40,
                            color='black', clip_on=False, transform=ax_top.transData)

        ax = fig.add_axes([0.10, 0.0, 0.7, 0.9])
        im = ax.imshow(z.to_numpy(dtype=np.float32), aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax, interpolation='none')

        cax = fig.add_axes([0.85, 0.08, 0.02, 0.2])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Expression (z-score)", fontsize=26)
        cbar.ax.tick_params(labelsize=26)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

        ax.set_xticks([])
        ax.set_yticks(np.arange(len(z.index)))
        ax.set_yticklabels(z.index, fontsize=40)
        ax.yaxis.tick_left()
        ax.tick_params(axis='y', labelleft=True, labelright=False)

        if len(order) > 1:
            change_points = np.where(group_labels.to_numpy()[:-1] != group_labels.to_numpy()[1:])[0]
            if len(change_points) > 0:
                split = change_points[0] + 1
                ax.axvline(split - 0.5, color='black', lw=2, linestyle='--')

        outdir = save_dir / 'cybersort_immune'
        save_panel(outdir / f"{filename}-{title}-genelist-heatmap.png", dpi=600, bbox_inches='tight')
        plt.close('all')


    # #Sialylation-related gene list
    # def sialytransferase_genes():
    #     sialytransferase = [
    #         "NPL",
    #         "ST3GAL1",
    #         "ST8SIA3",
    #         "ST3GAL2",
    #         "ST8SIA1",
    #         "ST3GAL5",
    #         "ST6GALNAC2",
    #         "ST6GALNAC4",
    #         "ST6GALNAC6",
    #         "CTSA",
    #         "ST8SIA6",
    #         "SLC35A1",
    #         "ST6GALNAC1",
    #         "GNE",
    #         "CMAS",
    #         "SLC17A5",
    #         "GLB1",
    #         "ST6GAL2",
    #         "ST3GAL6",
    #         "ST6GAL1",
    #         "NANP",
    #         "NEU3",
    #         "ST6GALNAC3",
    #         "NEU1",
    #         "NEU4",
    #         "NEU2",
    #         "ST3GAL4",
    #         "ST8SIA2",
    #         "ST6GALNAC5",
    #         "ST8SIA5",
    #         "ST3GAL3",
    #     ]
    #     return sialytransferase
    # -----------------------------------------------------------------------------
    # Main workflow
    # -----------------------------------------------------------------------------

    def main():

        mpl.rcParams['figure.figsize'] = [4, 3]
        mpl.rcParams['figure.dpi'] = 150
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['lines.linewidth'] = 1.

        # ---- 1. Load & QC raw expression -------------------------------------
        df_raw = load_raw_expression()
        df_sample = load_sample_metadata()
        df = merge_sample_metadata(df_raw, df_sample)


        df_kidney = subset_kidney_tissue(df)
        # Mitochondrial genes are NOT removed. ("no_mito" in the names below is
        # a historical label kept so the panel file names stay stable.)
        df_rcc = filter_expressed_genes(df_kidney)

        # ---- 2. Clinical annotation & ccRCC sample set ------------------------
        # all Normal samples + all ccRCC tumours with clinical annotation
        df_clin, df_ccrcc = load_clinical_data()
        df_ccrcc_no_mito_all = df_cancer_type(df_rcc, df_clin, df_ccrcc, match=False, ccRcc=True)
        dfs = [("df_ccrcc_no_mito_all", df_ccrcc_no_mito_all)]

        # ---- 3. Glycoprotein / glycogene-pathway gene lists --------------------
        glycoprotein_genes = build_glycoprotein_gene_list()
        df_gly_pathway, gly_pathway_genes = load_pathway_class_table()

        # Pathway classes with too few genes to be a meaningful "top pathway"
        # pick (e.g. Ceramidase/Hyaluronic acid/Sulfate synthase have only 3
        # genes each) -- excluded from the top-5/top-6 pathway rankings below
        # (Figure1 J, Figure3 I, SFigure1, SFigure4) so those slots backfill
        # from more broadly-supported pathways instead.
        pathway_gene_counts = df_gly_pathway.groupby('pathway class')['gene'].nunique()
        small_pathways = pathway_gene_counts[pathway_gene_counts < 5].index.tolist()

        # ---- 4. Glycosylation-based ccRCC subtyping (cluster I / II) -----------
        df_ccrcc_ca_all_clust, df_ccrcc_ca_gly_clust, df_ccrcc_nca_all_clust, df_ccrcc_nca_gly_clust = (
            build_ccrcc_clusters(dfs, gly_pathway_genes, save_dir=SAVE_DIR)
        )

        # primary working set: samples from the "no_mito, all-available" variant
        df_ca = df_ccrcc_ca_all_clust['df_ccrcc_no_mito_all']
        df_ca_gly = df_ccrcc_ca_gly_clust['df_ccrcc_no_mito_all']
        df_nca = df_ccrcc_nca_all_clust['df_ccrcc_no_mito_all']
        df_nca_gly = df_ccrcc_nca_gly_clust['df_ccrcc_no_mito_all']

        # ---- 5. Overall survival, cluster I vs. II (Figure3 B) -------------------
        overal_survival(df_ca, 'df_ccrcc_no_mito_all', save_dir=SAVE_DIR)

        # ---- 6. Pathway-level ANOVA scoring --------------------------------------
        df_ca_gly_pw = merge_pathway(df_ca_gly, df_gly_pathway)
        df_nca_gly_pw = merge_pathway(df_nca_gly, df_gly_pathway)

        df_ca_summary, df_ca_score = pathway_score(df_ca, 'cluster_type', df_gly_pathway)
        df_nca_summary, df_nca_score = pathway_score(df_nca, 'Type', df_gly_pathway)

        # ---- 7. Sorted pathway z-score heatmap + PCA ------------------------------
        df_ca_gly_pw_sort = pathway_sort(df_ca_gly_pw, gene=False)

        # all-pathway z-score matrix (samples ordered Subtype I then II), used
        # for the PCA panels below
        df_ca_clusted = pathway_zscore_matrix(df_ca_score, df_ca_gly_pw_sort, 23)
        # Figure3 C: taller, right-side pathway labels bigger, and the
        # gap between those labels and the colorbar closed (scoped to this call
        # only).
        plot_gly_pathway(df_ca_score, df_ca_gly_pw_sort, 10, 'subtype_by_glycogene_top10', save_dir=SAVE_DIR,
                          height_scale=1.38, fontsize_scale=1.3, cbar_dx=0.06)

        # Seaborn "ticks" theme (no grid, no top/right spines) for every panel
        # drawn from here on in this stage.
        sns.set(style="ticks", font_scale=0.8,
                rc={"axes.grid": False, "axes.spines.right": False, "axes.spines.top": False})

        df_nca_clusted = df_nca_gly_pw.set_index('gene').drop(columns='pathway class')

        # ---- 8. Per-gene scoring: mean-level t-test + per-sample z-scores ---------
        df_ca_with_clust = df_ca
        df_nca_with_clust = df_nca

        df_ca_gene_score = gene_score(df_ca_with_clust, df_gly_pathway, subtype=True)
        df_nca_gene_score = gene_score(df_nca_with_clust, df_gly_pathway, subtype=False)

        df_ca_gene_score_psample = gene_score_per_sample(df_ca_with_clust, df_gly_pathway, cancer=True)
        df_nca_gene_score_psample = gene_score_per_sample(df_nca_with_clust, df_gly_pathway, cancer=False)

        df_ca_gene_score['pathway class'] = df_ca_gene_score['pathway']
        df_nca_gene_score['pathway class'] = df_nca_gene_score['pathway']

        # Figure3 D: taller, everything bigger, legend on top and centered
        # (scoped to this call only).
        plot_gene_score(df_ca_score, df_ca_gene_score, 23, 'ccRCC_subtype_full', save_dir=SAVE_DIR,
                         height_scale=1.85, fontsize_scale=2.1, legend_dx=-0.2, legend_dy=-0.04,
                         width_scale=1.25)
        # Figure1 E: same treatment as Figure3 D above (scoped to this call only).
        plot_gene_score(df_nca_score, df_nca_gene_score, 23, 'ccRCC_norm_ca_full', save_dir=SAVE_DIR,
                         height_scale=1.55, fontsize_scale=1.6)

        # excluded from Figure3 G's own top-5-genes-per-pathway pick so it
        # doesn't duplicate genes already shown in the sialyltransferase family
        # heatmap (Figure5 D) -- not excluded from the Normal-vs-Cancer
        # version, which isn't paired with that panel.
        sialyltransferase_list = sialyltransferase_genes(df_gly_pathway)
        # Figure3 G: a little taller, everything a little bigger, colorbar text
        # bigger still (scoped to this call only -- Figure1 H shares this
        # function but wasn't asked for).
        top_genes_across_pathways_heatmap(df_ca_score, df_ca_gene_score, df_ca_gene_score_psample, k_pathways=5,
                                           genes_per_pathway=5, filename='ccRCC_subtype_top5pathways_genes',
                                           subtype=True, save_dir=SAVE_DIR, exclude_genes=sialyltransferase_list,
                                           exclude_pathways=small_pathways,
                                           height_scale=1.15, fontsize_scale=1.15, cbar_fontsize_scale=1.3)
        top_genes_across_pathways_heatmap(df_nca_score, df_nca_gene_score, df_nca_gene_score_psample, k_pathways=5,
                                           genes_per_pathway=5, filename='ccRCC_norm_ca_top5pathways_genes',
                                           subtype=False, save_dir=SAVE_DIR, cbar_offset=0.17,
                                           exclude_pathways=small_pathways)

        # ---- 9. Per-pathway PCA ---------------------------------------------------
        # Figure3 E and F (both from this same call, 'fc' and 'nlogp'
        # respectively): taller, everything bigger.
        plot_pca_pathway(df_ca_score, merge_pathway(df_ca_with_clust, df_gly_pathway), 23, 'ccRCC_subtype_full', df_gly_pathway, subtype=True, save_dir=SAVE_DIR,
                          value_scale={'fc': (1.45, 1.5), 'nlogp': (1.45, 1.5)}, values=('nlogp', 'fc'))
        # Figure1 F ('fc' only -- G, the 'nlogp' sibling from this same call,
        # shares a different row and wasn't asked to change): a little taller,
        # everything a little bigger.
        plot_pca_pathway(df_nca_score, merge_pathway(df_nca_with_clust, df_gly_pathway), 23, 'ccRCC_norm_ca_full', df_gly_pathway, subtype=False, save_dir=SAVE_DIR,
                          value_scale={'fc': (1.2, 1.3)}, values=('nlogp', 'fc'))

        # Pathway classes with <5 genes (`small_pathways`) are excluded from
        # these top-pathway rankings -- e.g. 'Sulfate synthase' only has 3
        # genes, so its row was mostly blank (5 of 6 PC-pair panels hidden);
        # the k slots backfill from the next-ranked, more broadly-supported
        # pathways instead.
        # SFigure1/4: 7 pathways now (was 6) -- filename kept as "top6" so
        # nothing downstream that reads it by that literal name needs to change.
        PCA_plot_grid_full(df_ca_score, df_ca_clusted, 7, df_gly_pathway, 'PCs_top6_pathways_subtype_full_combined', subtype=True, save_dir=SAVE_DIR,
                           exclude_pathways=small_pathways)
        PCA_plot_grid_full(df_nca_score, df_nca_clusted, 7, df_gly_pathway, 'PCs_top6_pathways_norm_ca_full_combined', subtype=False, save_dir=SAVE_DIR,
                           exclude_pathways=small_pathways)

        # Figure1 J: full-width, one-row overview strip.
        PCA_overview_top_pathways_row(df_nca_score, df_nca_clusted, df_gly_pathway, subtype=False, k=5,
                                       overall_title='All glycogenes',
                                       filename='PCs_overview_top5_pathways_norm_ca_1row', save_dir=SAVE_DIR,
                                       exclude_pathways=small_pathways)
        # Figure3 I: shares a row with F (2 rows x 3 cols), not its own
        # full-width row.
        PCA_overview_top_pathways_grid(df_ca_score, df_ca_clusted, df_gly_pathway, subtype=True, k=5,
                                        overall_title='All glycogenes',
                                        filename='PCs_overview_top5_pathways_subtype', save_dir=SAVE_DIR,
                                        exclude_pathways=small_pathways)

        # genes ordered most- to least-significant (top to bottom) per comparison,
        # reusing the per-gene Welch t-test p-values already computed above.
        sialyltransferase_subtype_order = (
            df_ca_gene_score[df_ca_gene_score['gene'].isin(sialyltransferase_list)]
            .sort_values('p')['gene'].tolist()
        )
        sialyltransferase_nca_order = (
            df_nca_gene_score[df_nca_gene_score['gene'].isin(sialyltransferase_list)]
            .sort_values('p')['gene'].tolist()
        )
        plot_gene_list_zscore_heatmap(sialyltransferase_subtype_order, 'sialyltransferase_genes_subtype', df_ca_clusted, df_gly_pathway, subtype=True, save_dir=SAVE_DIR)
        plot_gene_list_zscore_heatmap(sialyltransferase_nca_order, 'sialyltransferase_genes_norm_ca', df_nca_clusted, df_gly_pathway, subtype=False, save_dir=SAVE_DIR)

        # ---- 10. Save cluster-annotated expression tables (shared data) -----------
        df_ccrcc_ca_glob = df_ca
        df_ccrcc_nca_glob = df_nca


        # ---- 11. CIBERSORTx immune-fraction integration ----------------------------
        df_na_immune = df_immune_merge('na_fraction', df_ccrcc_no_mito_all, cancer=False)
        df_ca_immune = df_immune_merge('ca_fraction', df_ccrcc_no_mito_all, cancer=True)

        df_nca_immune = pd.concat([df_na_immune, df_ca_immune], axis=0, ignore_index=True)

        df_count_cluster_type = df_ca_with_clust[['Count_x', 'cluster_type']].drop_duplicates()
        df_ca_immune_type = pd.merge(df_ca_immune, df_count_cluster_type, on='Count_x', how='inner')

        # Normal-vs-Cancer table: the Type column is already present and
        # normal samples have no subtype, so no further merge is needed.
        df_nca_immune_type = df_nca_immune

        plot_mean_immune(df_ca_immune_type, 'ccRCC_subtype', cancer=True, save_dir=SAVE_DIR)
        plot_mean_immune(df_nca_immune_type, 'ccRCC_na_ca', cancer=False, save_dir=SAVE_DIR)

        plot_ind_immune(df_ca_immune_type, 'ccRCC-subtype', cancer=True, save_dir=SAVE_DIR)
        plot_ind_immune(df_nca_immune_type, 'ccRCC-na-ca', cancer=False, save_dir=SAVE_DIR)

        # ---- 12. Gene-of-interest & custom gene-list heatmaps -----------------------
        gene_interest_heatmap(df_ccrcc_nca_glob, 'ccRCC_na_ca', 'siglec', cancer=False, save_dir=SAVE_DIR)
        gene_interest_heatmap(df_ccrcc_ca_glob, 'ccRCC_subtype2', 'siglec', cancer=True, save_dir=SAVE_DIR,
                              save_mean=False)

        df_interest_genes = pd.read_excel(DATA_DIR / 'immune_regulated_genes.xlsx')
        immune_list = df_interest_genes['Gene'].to_list()

        genelist_heatmap(df_ccrcc_ca_glob, 'immune gene', 'ccRCC_subtype_20', immune_list, k=20, cancer=True, save_dir=SAVE_DIR)
        genelist_heatmap(df_ccrcc_nca_glob, 'immune gene', 'ccRCC_nca_20', immune_list, k=20, cancer=False, save_dir=SAVE_DIR)

        # data handed to the later stages (in memory)
        return {
            'df_ca': trim_cluster_table(df_ccrcc_ca_glob),      # stage_03
            'df_nca': trim_cluster_table(df_ccrcc_nca_glob),    # stage_02
            'glycoprotein_genes': glycoprotein_genes,           # stage_02
            'immune_ca': df_ca_immune_type,                     # stage_04
            'immune_nca': df_nca_immune,                        # stage_04
            'gene_score_psample_ca': df_ca_gene_score_psample,  # stage_04
            'gene_score_psample_nca': df_nca_gene_score_psample,  # stage_04
        }


    return main()

# =============================================================================
# stage_02_ccrcc_nca_volcano
# =============================================================================

def stage_02_ccrcc_nca_volcano(df_nca, glycoprotein_genes):
    """Normal vs. Cancer (nca) differential expression in ccRCC: volcano plots,
    DEG tables and supervised heatmaps for all genes, glycoprotein genes and
    glycogenes. Reads the cluster-annotated tables and the glycoprotein gene
    list from stage_01. Panels are rendered in memory.
    """
    _reset_mpl_defaults()

    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.patches import Patch
    import scipy as sp
    import seaborn as sns
    from adjustText import adjust_text

    SCRIPT_NAME = "2026_ccrcc_nca_volcano"
    OUTPUT_DIR = RESULT_MANUSCRIPT_DIR / SCRIPT_NAME
    SAVE_DIR = OUTPUT_DIR

    # Upregulated/downregulated (volcano coloring, save_degs) requires FDR < 0.05
    # (Benjamini-Hochberg corrected) and this effect-size floor.
    MIN_ABS_LOG2FC = 0.5


    # -----------------------------------------------------------------------------
    # General plotting helpers
    # -----------------------------------------------------------------------------

    class LoggedDataLocator(mpl.ticker.Locator):
        """Set exponential tick marks for data that has already been logged."""

        def __init__(self, base=10, skip_base=False):
            self._base = base
            self._skip_base = skip_base

        def set_params(self, base):
            if base is not None:
                self._base = base

        def __call__(self):
            vmin, vmax = self.axis.get_view_interval()
            return self.tick_values(vmin, vmax)

        def tick_values(self, vmin, vmax):
            if vmax < vmin:
                vmin, vmax = vmax, vmin

            locs = np.array(
                [
                    self._base ** c * range((2 if self._skip_base else 1), self._base)
                    for c in np.arange(np.floor(vmin // 1), np.ceil(vmax / 1))
                ]
            ).flatten()
            locs = locs[(locs > self._base ** vmin) & (locs < self._base ** vmax)]
            return np.log(locs) / np.log(self._base)

        def view_limits(self, dmin, dmax):
            return (
                self._base ** (dmin // 1) * np.floor(self._base ** ((dmin % 1))),
                self._base ** (dmax // 1) * np.ceil(self._base ** ((dmax % 1))),
            )


    def label_plot(ax, x, y, labels, fontsize=8, arrowprops=None, plot_kwargs=None, text_kwargs=None):
        if plot_kwargs is None:
            plot_kwargs = {
                'marker': 'o',
                'linestyle': 'None',
                'mfc': 'None',
                'mec': 'r',
                'ms': 4,
            }

        if text_kwargs is None:
            text_kwargs = {
                'bbox': dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7)
            }

        ax.plot(x, y, **plot_kwargs)

        texts = []
        for x_, y_, label_ in zip(x, y, labels):
            texts.append(
                ax.text(
                    x_,
                    y_,
                    label_,
                    fontsize=fontsize,
                    ha='center',
                    va='center',
                    zorder=9,
                    **text_kwargs,
                )
            )

        adjust_text(
            texts,
            x=x,
            y=y,
            ax=ax,
            arrowprops=arrowprops,
            expand_text=(1.8, 1.8),
            expand_points=(1.5, 1.5),
            force_text=(2.5, 2.5),
            force_points=(2.0, 2.0),
            force_objects=(2.0, 2.0),
            lim=500,
            only_move={'text': 'xy', 'static': 'xy', 'explode': 'xy', 'pull': 'xy'},
        )


    # -----------------------------------------------------------------------------
    # Notebook-derived analysis functions
    # -----------------------------------------------------------------------------

    def df_isin_gene(df, gene_table):
        genes = gene_table['gene'].to_list()
        return df[df['gene'].isin(genes)]


    def ab_test_plot(df, title, gly=True, cancer=True, alpha=0.05, min_cutoff=10.0, plot_results=True,
                      plot_title=None):
        col = rw_function.desired_order(cancer)
        col_t = rw_function.col_type(cancer)
        vals = ['log_tpm', 'transcripts_per_million']

        df_A = pd.pivot_table(df[df[col_t] == col[0]], index='sample_id', columns='gene', values=vals[0])
        df_B = pd.pivot_table(df[df[col_t] == col[1]], index='sample_id', columns='gene', values=vals[0])

        n_genes = df_A.shape[1]
        print(f"there are {n_genes} expressed genes")
        assert np.all(df_A.columns == df_B.columns), "Gene names don't match between A and B"

        _, p = sp.stats.ttest_ind(np.log10(1 + df_A), np.log10(1 + df_B), equal_var=True)

        with np.errstate(divide='ignore'):
            log_fold_change = np.log2(df_B.mean() / df_A.mean())
            log_fold_change_extremum = 5 * np.log2(10)
            log_fold_change[df_B.mean() == 0] = log_fold_change_extremum
            log_fold_change[df_A.mean() == 0] = -log_fold_change_extremum

        rank = np.argsort(np.argsort(p))
        alpha_Holm_Bonferroni = alpha / (len(p) - rank + 1)
        alpha_Benjamini_Hochberg = alpha * (rank + 1) / len(p)

        if np.any(p < alpha_Benjamini_Hochberg):
            alpha_Benjamini_Hochberg_cutoff = np.max(p[p <= alpha_Benjamini_Hochberg])
        else:
            alpha_Benjamini_Hochberg_cutoff = np.min(alpha_Benjamini_Hochberg)

        # Upregulated/downregulated (red/blue) requires both FDR significance
        # (BH-corrected p) and an effect-size floor, |log2 fold change| > 0.5 --
        # matching the criteria used downstream for DEG selection (save_degs).
        # Everything else (including FDR-significant genes with a small effect
        # size) falls back to the density-colored background scatter, rather
        # than being dropped from the plot entirely.
        is_sig = (p <= alpha_Benjamini_Hochberg_cutoff) & (np.abs(log_fold_change) > MIN_ABS_LOG2FC)

        rw_function.scatterplot_kde_norm(
            log_fold_change[~is_sig],
            -np.log10(p[~is_sig]),
            zorder=2,
        )

        plt.scatter(
            log_fold_change[is_sig & (log_fold_change > 0)],
            -np.log10(p[is_sig & (log_fold_change > 0)]),
            c='r',
            s=0.5,
            zorder=3,
        )

        plt.scatter(
            log_fold_change[is_sig & (log_fold_change < 0)],
            -np.log10(p[is_sig & (log_fold_change < 0)]),
            c='b',
            s=0.5,
            zorder=3,
        )

        print('Number of DEGs (Holm-Bonferroni):', sum(p <= alpha_Holm_Bonferroni))
        print('Number of DEGs (Benjamini-Hochberg):', sum(p <= alpha_Benjamini_Hochberg_cutoff))

        plt.plot(
            [-log_fold_change_extremum, log_fold_change_extremum],
            -np.log10([alpha_Benjamini_Hochberg_cutoff, alpha_Benjamini_Hochberg_cutoff]),
            c='0.8',
            lw=2,
            zorder=1,
        )
        axis_label_fontsize = 17
        if plot_title is not None:
            plt.title(plot_title, fontsize=axis_label_fontsize)
        plt.xlabel('log$_2$ fold change', fontsize=axis_label_fontsize)
        plt.ylabel('-log$_{10}$p', fontsize=axis_label_fontsize)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))

        rg = (-1.5, 1.5) if cancer else (-1, 1)
        if gly:
            rg = (-0.7, 0.7)
        rg = (rg[0] * np.log2(10), rg[1] * np.log2(10))

        plt.xlim(*rg)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))

        plot_name = f"{title}_volcano"
        filepath = SAVE_DIR / f"{plot_name}.png"

        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-300, 1.0)

        df_ = pd.DataFrame(
            {
                'gene': df_A.columns,
                'log_fold_change': log_fold_change,
                'fold_change': 2 ** log_fold_change,
                f'tpm_{col[0]}': df_A.mean(),
                f'tpm_{col[1]}': df_B.mean(),
                'p_uncorrected': p,
                'p_corrected': 1 - (1 - p) ** n_genes,
                'log2_p': -np.log10(p),
                'alpha_Holm_Bonferroni': alpha_Holm_Bonferroni,
                'alpha_Benjamini_Hochberg': alpha_Benjamini_Hochberg,
                'alpha_Benjamini_Hochberg_cutoff': alpha_Benjamini_Hochberg_cutoff,
                'mean': df_A.mean(),
                'var': df_A.var(axis=0),
                't': np.nan,
            },
            index=df_A.columns,
        )

        df_label = df_.dropna(subset=['gene', 'log_fold_change', 'p_uncorrected', 'p_corrected']).copy()
        df_label = df_label[df_label['p_corrected'] < 1e-2]

        df_p = df_label.sort_values('p_uncorrected', ascending=True).head(8)
        df_fc = df_label.reindex(df_label['log_fold_change'].abs().sort_values(ascending=False).index).head(5)
        df_fn1 = df_label[df_label['gene'].str.upper() == 'FN1']
        df_label = pd.concat([df_p, df_fc, df_fn1]).drop_duplicates(subset='gene')

        label_plot(
            plt.gca(),
            df_label['log_fold_change'].to_numpy(),
            -np.log10(df_label['p_uncorrected']).to_numpy(),
            df_label['gene'].tolist(),
            fontsize=10,
            arrowprops=dict(arrowstyle='-', color=None, lw=0.5),
        )

        save_panel(filepath, bbox_inches='tight', dpi=600)
        plt.close()
        return df_


    def deg_table(df_abtest):
        """Upregulated/downregulated DEGs: FDR < 0.05 (Benjamini-Hochberg
        corrected p) and |log2 fold change| > `MIN_ABS_LOG2FC`, matching the
        volcano plot's red/blue coloring in `ab_test_plot`."""
        sig = df_abtest[(df_abtest['p_uncorrected'] <= df_abtest['alpha_Benjamini_Hochberg_cutoff'])
                         & (df_abtest['log_fold_change'].abs() > MIN_ABS_LOG2FC)].copy()
        sig = sig.reset_index(drop=True)[['gene', 'log_fold_change', 'fold_change', 'p_uncorrected', 'p_corrected']].copy()
        sig['regulation'] = sig['log_fold_change'].apply(lambda x: 'upregulated' if x > 0 else 'downregulated')
        sig = sig.rename(columns={'log_fold_change': 'log2_fold_change'})
        return sig


    def cluster_heatmap(df_, title, order=True, cancer=True, height_scale=1.0, fontsize_scale=1.0):
        """`height_scale`/`fontsize_scale` scale this panel's height and every
        text element (default 1.0, unchanged)."""
        df_ = df_.copy()

        type_col1 = rw_function.col_type(cancer)
        type_col = rw_function.find_col(df_, type_col1)
        stage_col = rw_function.find_col(df_, 'Path Stage')
        sample_col = rw_function.find_col(df_, 'Count')
        gene_col = rw_function.find_col(df_, 'gene')
        value_col = rw_function.find_col(df_, 'log_tpm')

        df_['sample number'] = df_[sample_col]
        df_['sample_label'] = df_['sample number'].astype(str) + ' (' + df_[type_col].astype(str) + ')'

        df_2 = df_.pivot_table(index=gene_col, columns='sample_label', values=value_col, aggfunc='mean')

        if order:
            columns = (
                df_.sort_values([type_col, stage_col, 'sample number'], ascending=[False, True, True])['sample_label']
                .drop_duplicates()
                .tolist()
            )
            df_2 = df_2.reindex(columns=columns)

        row_mean = df_2.mean(axis=1)
        row_std = df_2.std(axis=1, ddof=0)
        df_z = df_2.sub(row_mean, axis=0).div(row_std.replace(0, 1), axis=0)
        flat_mask = row_std.eq(0)
        df_z.loc[flat_mask] = 0

        sample_type_map = (
            df_[['sample_label', type_col]].drop_duplicates().set_index('sample_label')[type_col].to_dict()
        )

        if cancer:
            group_colors = {'I': 'red', 'II': 'blue'}
            group_labels = {'I': 'Subtype I', 'II': 'Subtype II'}
        else:
            group_colors = {'Normal': 'black', 'Cancer': 'red'}
            group_labels = {'Normal': 'Normal', 'Cancer': 'Cancer'}
        col_colors = [group_colors[str(sample_type_map[s]).strip()] for s in df_z.columns]

        plt.close()

        g = sns.clustermap(
            df_z,
            cmap='bwr',
            figsize=(10, 10 * height_scale),
            method='ward',
            metric='euclidean',
            col_cluster=not order,
            row_cluster=True,
            col_colors=col_colors,
            xticklabels=False,
            yticklabels=False,
            cbar_pos=(1, .15, .03, .1),
        )

        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=20 * fontsize_scale)
        g.ax_heatmap.set_xlabel('')
        g.ax_heatmap.set_ylabel('')
        g.cax.tick_params(labelsize=26 * fontsize_scale)
        g.cax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        g.cax.set_ylabel('z-score', fontsize=26 * fontsize_scale)

        if order:
            # Samples are kept in fixed group order (one contiguous block per
            # group), so a top group-header label, centered over each block,
            # unambiguously identifies it.
            group_of_col = [str(sample_type_map[s]).strip() for s in df_z.columns]
            n_cols = len(group_of_col)
            seg_start = 0
            for i in range(1, n_cols + 1):
                if i == n_cols or group_of_col[i] != group_of_col[seg_start]:
                    grp = group_of_col[seg_start]
                    label = group_labels[grp]
                    center_frac = (seg_start + i) / (2 * n_cols)
                    g.ax_col_colors.text(center_frac, 1.4, label, ha='center', va='bottom',
                                          fontsize=40 * fontsize_scale, fontweight='bold', clip_on=False,
                                          transform=g.ax_col_colors.transAxes)
                    seg_start = i
        else:
            # Columns are hierarchically clustered (`col_cluster=True`), so
            # groups can end up interleaved rather than in one contiguous run
            # each -- a top header label per run would be misleading (or, with
            # many short runs, an unreadable overlapping mess). Use a color
            # legend instead of per-segment labels.
            handles = [Patch(facecolor=group_colors[grp], label=group_labels[grp]) for grp in group_colors]
            g.ax_heatmap.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.90, 1.05),
                                 fontsize=26 * fontsize_scale, frameon=False, handlelength=1.2, handleheight=1.2)

        plot_name = f"{title}_heatmap"
        filepath = SAVE_DIR / f'{plot_name}.png'
        save_panel(filepath, bbox_inches='tight', dpi=600)
        plt.close()


    def supervised_heatmap(p, df_kidney, k, title, order=True, cancer=True, height_scale=1.0, fontsize_scale=1.0):
        p_sub = p.dropna(subset=['gene', 'p_uncorrected', 'log_fold_change']).copy()
        p_sub = p_sub[p_sub['p_uncorrected'] < 1e-2]
        p_sub = p_sub.sort_values('p_uncorrected', ascending=True)

        gene_top = p_sub.head(k)
        # p_uncorrected cutoff implied by taking exactly the top-k genes by
        # p-value rank (the largest p_uncorrected among the selected genes).
        cutoff_p = gene_top['p_uncorrected'].max() if len(gene_top) > 0 else float('nan')
        print(f"{title}: top {k} genes -> p_uncorrected cutoff <= {cutoff_p:.3g} "
              f"(of {len(p_sub)} genes with p_uncorrected < 1e-2)")

        target_genes = gene_top['gene']
        df_ = df_kidney[df_kidney['gene'].isin(target_genes)].copy()
        cluster_heatmap(df_, title=title, order=order, cancer=cancer,
                         height_scale=height_scale, fontsize_scale=fontsize_scale)


    # -----------------------------------------------------------------------------
    # Main workflow
    # -----------------------------------------------------------------------------

    def main():

        mpl.rcParams['figure.figsize'] = [4, 3]
        mpl.rcParams['figure.dpi'] = 150
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['lines.linewidth'] = 1.0

        df_glycogene = read_gene_list('glycogene_list_430.csv')
        df_gprots = pd.DataFrame({'gene': glycoprotein_genes})

        df_nca_gprots = df_isin_gene(df_nca, df_gprots)
        df_nca_glycogene = df_isin_gene(df_nca, df_glycogene)

        # Figure1 A-C
        ab_test_plot(df_nca_glycogene, 'glycogene_nca-log10', gly=True, cancer=False, alpha=0.05, min_cutoff=1.0, plot_title='Glycogenes')

        df_nca_abtest = ab_test_plot(df_nca, 'nca_glob-log', gly=False, cancer=False, alpha=0.05, min_cutoff=1.0, plot_title='Global genes')
        df_nca_g_prots_abtest = ab_test_plot(df_nca_gprots, 'glycoproteins_nca_log', gly=False, cancer=False, alpha=0.05, min_cutoff=1.0, plot_title='Glycan_based genes')

        # Figure1 D: a little taller, everything a little bigger (scoped to
        # this call only).
        supervised_heatmap(df_nca_abtest, df_nca, 1500, 'nca_1500', order=False, cancer=False,
                            height_scale=1.2, fontsize_scale=1.3)

        # for stage_05 (GO enrichment / GSEA): significant DEGs and every tested gene
        return {
            'global': (deg_table(df_nca_abtest),
                       df_nca_abtest.rename(columns={'log_fold_change': 'log2_fold_change'})),
            'glycoprotein': (deg_table(df_nca_g_prots_abtest),
                             df_nca_g_prots_abtest.rename(columns={'log_fold_change': 'log2_fold_change'})),
        }


    return main()

# =============================================================================
# stage_03_ccrcc_survival
# =============================================================================

def stage_03_ccrcc_survival(df_ca):
    """Kaplan-Meier / log-rank screen of SIGLEC genes in ccRCC tumours (median
    split of log TPM into High/Low), using the cluster-annotated table from
    stage_01. Panels are rendered in memory.
    (Survival by glycosylation subtype is plotted in stage_01.)
    """
    _reset_mpl_defaults()

    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import math
    import re
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib.lines import Line2D

    SCRIPT_NAME = "2026_ccRcc_survival"
    OUTPUT_DIR = RESULT_MANUSCRIPT_DIR / SCRIPT_NAME
    SAVE_DIR = OUTPUT_DIR

    mpl.rcParams.update({
        "font.size": 40,
        "axes.titlesize": 40,
        "axes.labelsize": 40,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
        "legend.fontsize": 40,
        "legend.frameon": False,
        "legend.framealpha": 0.0,
        "legend.facecolor": "none",
        "figure.dpi": 300,
    })


    def find_col(df, base_name):
        if base_name in df.columns:
            return base_name
        elif f'{base_name}_x' in df.columns:
            return f'{base_name}_x'
        elif f"{base_name}_y" in df.columns:
            return f"{base_name}_y"
        return None


    def generate_survival_plots() -> None:
        """Kaplan-Meier screen of SIGLEC genes in ccRCC tumours, saved under SAVE_DIR."""
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        # identify SIGLEC genes
        genes = df_ca['gene'].dropna().unique().tolist()
        def sort_key(x):
            m = re.search(r'SIGLEC(\d+)', x)
            if m:
                return (0, int(m.group(1)), x)
            return (1, float('inf'), x)
        siglec = sorted([g for g in genes if 'SIGLEC' in str(g)], key=sort_key)

        def km_logrank_screen_plot(df_, gene_list, title,
                                    p_cutoff=0.05, min_n=10, min_group=5,
                                    ncol=3, figsize_scale=(4, 3),
                                    title_fontsize=40, label_fontsize=40,
                                    tick_fontsize=40, line_width=3):

            sample_col = find_col(df_, 'sample_id') or 'sample_id'
            gene_col = find_col(df_, 'gene') or 'gene'
            expr_col = find_col(df_, 'log_tpm') or 'log_tpm'
            event_col = find_col(df_, 'Dead=1. Alive=0')
            death_time_col = find_col(df_, 'days to death')
            followup_time_col = find_col(df_, 'Days to last F/U')

            # reshape
            expr = df_.pivot_table(index=sample_col, columns=gene_col, values=expr_col, aggfunc='mean')
            clin_cols = [sample_col, event_col, death_time_col, followup_time_col]
            clin = df_[clin_cols].drop_duplicates(sample_col).set_index(sample_col)
            surv = clin.join(expr)
            surv['time'] = np.where(surv[event_col] == 1, surv[death_time_col], surv[followup_time_col])
            surv = surv.dropna(subset=['time', event_col])

            # per-gene tests
            hits = []
            for g in gene_list:
                if g not in surv.columns:
                    continue
                sub = surv[['time', event_col, g]].dropna()
                if sub.shape[0] < min_n:
                    continue
                cut = sub[g].median()
                hi = sub[sub[g] >= cut]
                lo = sub[sub[g] < cut]
                if min(len(hi), len(lo)) < min_group:
                    continue
                lr = logrank_test(hi['time'], lo['time'], event_observed_A=hi[event_col], event_observed_B=lo[event_col])
                if lr.p_value < p_cutoff:
                    hits.append({'gene': g, 'p_value': lr.p_value, 'chi2': lr.test_statistic})

            hits_df = pd.DataFrame(hits)
            if hits_df.empty:
                print(f"No significant genes at p < {p_cutoff} for {title}")
                return hits_df

            # keep gene_list's own order (already SIGLEC-number sorted by the
            # caller) rather than re-sorting by significance.
            gene_rank = {g: i for i, g in enumerate(gene_list)}
            hits_df = hits_df.sort_values(by='gene', key=lambda s: s.map(gene_rank)).reset_index(drop=True)

            # compute legend counts (High/Low) from the first significant gene so
            # they can be shown in the blank-area legend
            legend_n_high = 0
            legend_n_low = 0
            if not hits_df.empty:
                first_gene = hits_df.iloc[0]['gene']
                if first_gene in surv.columns:
                    sub0 = surv[['time', event_col, first_gene]].dropna().copy()
                    cut0 = sub0[first_gene].median()
                    legend_n_high = int((sub0[first_gene] >= cut0).sum())
                    legend_n_low = int((sub0[first_gene] < cut0).sum())

            # multipanel KM plot
            n = len(hits_df)
            nrow = math.ceil(n / ncol)
            fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * figsize_scale[0], nrow * figsize_scale[1]))
            if n == 1:
                ax = np.array([ax])
            else:
                ax = np.array(ax).flatten()
            kmf = KaplanMeierFitter()
            for i, row in enumerate(hits_df.itertuples(index=False)):
                g = row.gene
                sub = surv[['time', event_col, g]].dropna().copy()
                cut = sub[g].median()
                sub['expr_group'] = np.where(sub[g] >= cut, 'High', 'Low')
                for label, grp in sub.groupby('expr_group'):
                    colour = 'red' if label.lower().startswith('high') else 'black'
                    kmf.fit(grp['time'], event_observed=grp[event_col], label=label)
                    # avoid adding per-axis legends by passing label=None to the plot call
                    kmf.plot_survival_function(ax=ax[i], color=colour, ci_show=False, linewidth=line_width, label=None)
                # annotate counts for High/Low groups (counts moved to blank-area legend)
                n_high = int((sub['expr_group'] == 'High').sum())
                n_low = int((sub['expr_group'] == 'Low').sum())
                ax[i].set_title(f"{g}\np={row.p_value:.3g}", fontsize=title_fontsize)
                # x-axis in days with exactly 3 ticks (start, middle, end)
                ax[i].set_ylabel("Survival", fontsize=label_fontsize)
                try:
                    max_time = float(sub['time'].max())
                except Exception:
                    max_time = float(ax[i].get_xlim()[1])
                # fixed ticks at 0, 3000, 6000 days
                new_ticks = np.array([0, 3000, 6000])
                ax[i].set_xticks(new_ticks)
                day_labels = [str(int(t)) for t in new_ticks]
                ax[i].set_xticklabels(day_labels)
                ax[i].set_xlabel("Days", fontsize=label_fontsize)
                ax[i].tick_params(axis='both', labelsize=tick_fontsize)
                for spine in ax[i].spines.values():
                    spine.set_linewidth(1.5)
            # ensure no per-axis legends remain
            for a in ax:
                leg = a.get_legend()
                if leg is not None:
                    try:
                        leg.remove()
                    except Exception:
                        pass
            # create consistent legend handles (High=red, Low=black)
            handles = [Line2D([0], [0], color='red', lw=line_width), Line2D([0], [0], color='black', lw=line_width)]
            labels = [f'High (n={legend_n_high})', f'Low (n={legend_n_low})']
            # prefer placing legend into the first unused subplot (blank area)
            unused_indices = [j for j in range(len(ax)) if j > i]
            legend_placed = False
            if len(unused_indices) > 0:
                legend_idx = unused_indices[0]
                try:
                    legend_ax = ax[legend_idx]
                    # keep this axis blank and use it for the legend
                    legend_ax.clear()
                    legend_ax.axis('off')
                    legend_ax.legend(handles, labels, fontsize=36, frameon=False, loc='center',
                                      bbox_to_anchor=(0.5, 0.65))
                    legend_placed = True
                except Exception:
                    legend_placed = False
            if not legend_placed:
                # fallback for a grid with no blank cell at all (e.g. exactly
                # ncol x nrow hits): put the legend at the very bottom, in one
                # line -- a legend to the right would compete with the subplots
                # for width when they're square, at a fixed overall page width.
                try:
                    fig.legend(handles, labels, fontsize=36, frameon=False, ncol=2,
                               loc='upper center', bbox_to_anchor=(0.5, -0.02))
                    legend_placed = True
                except Exception:
                    legend_placed = False
            # hide any remaining unused subplots, but keep legend axis visible
            for j in range(i + 1, len(ax)):
                if 'legend_idx' in locals() and j == legend_idx:
                    continue
                ax[j].set_visible(False)
            fig.subplots_adjust(wspace=0.7, hspace=1.3)

            out_dir = SAVE_DIR
            save_panel(out_dir / f'{title}-survival.PNG', bbox_inches='tight', dpi=600)
            plt.close(fig)

        # SFigure/Figure panels: significant genes only (p < 0.05) ...
        km_logrank_screen_plot(df_ca, siglec, "ca-only-Siglec-top_p", p_cutoff=0.05, ncol=5, figsize_scale=(10.3, 6.8), title_fontsize=40, label_fontsize=40, tick_fontsize=40, line_width=3)
        # ... and every SIGLEC gene with enough data (min_n/min_group): a
        # p_cutoff above 1 lets every p-value through, including exactly 1.0.
        km_logrank_screen_plot(df_ca, siglec, "ca-only-Siglec-all", p_cutoff=1.01, ncol=4, figsize_scale=(6, 6), title_fontsize=40, label_fontsize=40, tick_fontsize=40, line_width=3)


    generate_survival_plots()
    print("SIGLEC survival panels done")

# =============================================================================
# stage_04_cybertx_nca_immune_glyco_full
# =============================================================================

def stage_04_cybertx_nca_immune_glyco_full(immune_ca, immune_nca, gene_score_psample_ca, gene_score_psample_nca):
    """CIBERSORTx immune fractions vs. glycosylation-pathway scores, for both
    Normal vs. Cancer and Subtype I vs. II. Panels are rendered in memory.

    It uses two tables from stage_01:
    - the per-sample, per-gene glycogene z-scores, averaged here to a per-sample,
      per-pathway-class score;
    - the CIBERSORTx immune-cell fraction tables merged with cluster/Type labels


    Every function takes a `cancer` flag and uses
    `rw_function.col_type`/`desired_order` to generalize across both the
    ccRCC glycosylation-subtype comparison (cluster I vs. II) and the
    Normal-vs-Cancer comparison.

    Pipeline summary / what each plot shows
    ----------------------------------------
    1.  `immune_group_summary` + `immune_barplot_summary` -- mean immune-cell
        fraction difference between the two groups, as a horizontal bar chart
        with significance stars (Welch t-test, BH-FDR) (Figure2 B, Figure4 C).
    2.  `corr_for_groups` + `plot_pathway_vs_immune_by_group` -- side-by-side
        heatmaps (one panel per group) of pathway-class x immune-cell Spearman
        correlation, annotated with FDR significance stars (SFigure2 A,
        SFigure5 B).
    3.  `plot_difference` -- a single heatmap of the correlation difference
        between the two groups (group2 - group1) (Figure2 C, Figure4 D).
    """
    _reset_mpl_defaults()

    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    np.long = np.int64

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import seaborn as sns

    from scipy.stats import spearmanr, ttest_ind
    from statsmodels.stats.multitest import multipletests

    SCRIPT_NAME = "2026_cybertx_nca_immune_glyco_full"
    OUTPUT_DIR = RESULT_MANUSCRIPT_DIR / SCRIPT_NAME
    SAVE_DIR = OUTPUT_DIR


    # -----------------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------------

    def df_gly_path(df_glyco_path_nca):
        """Add a `Mixture` column (= `sample`) so gene-score tables can be
        merged with the CIBERSORTx tables on a common sample key."""
        df_glyco_path = df_glyco_path_nca.copy()
        df_glyco_path['Mixture'] = df_glyco_path['sample']
        return df_glyco_path


    # -----------------------------------------------------------------------------
    # Mean immune-cell fraction difference between groups (bar plot)
    # -----------------------------------------------------------------------------

    def immune_group_summary(df_immune, df_glyco_path, title, cancer=True,
                              mixture_col='Mixture', save_dir=SAVE_DIR):
        """Per immune-cell-type mean fraction in each group, their difference,
        and a Welch t-test (BH-FDR) between groups."""
        group_base = rw_function.col_type(cancer)
        order = rw_function.desired_order(cancer)

        type_map = df_glyco_path[[mixture_col, group_base]].dropna().copy()
        type_map[group_base] = type_map[group_base].astype(str).str.strip()
        type_map = (
            type_map.groupby(mixture_col)[group_base]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
            .reset_index()
        )

        merged = df_immune.merge(type_map, on=mixture_col, how='inner').copy()
        group_col = rw_function.find_col(merged, group_base)

        merged[group_col] = merged[group_col].astype(str).str.strip()
        merged = merged[merged[group_col].isin(order)].copy()
        if merged.empty:
            raise ValueError(f"No rows left after merge for groups {order}")

        immune_cols = [c for c in merged.columns if c not in [mixture_col, group_col]]
        merged[immune_cols] = merged[immune_cols].apply(pd.to_numeric, errors='coerce')

        g1, g2 = order[0], order[1]
        means = merged.groupby(group_col)[immune_cols].mean()
        for grp in order:
            if grp not in means.index:
                raise ValueError(f"No rows for group '{grp}' after merge.")

        delta = means.loc[g2] - means.loc[g1]

        pvals, n1, n2 = [], [], []
        for col in immune_cols:
            a = merged.loc[merged[group_col] == g1, col].dropna()
            b = merged.loc[merged[group_col] == g2, col].dropna()
            n1.append(len(a))
            n2.append(len(b))
            p = np.nan if (len(a) < 2 or len(b) < 2) else ttest_ind(a, b, equal_var=False).pvalue
            pvals.append(p)

        pvals = np.array(pvals, dtype=float)
        qvals = np.full_like(pvals, np.nan, dtype=float)
        mask = ~np.isnan(pvals)
        if mask.any():
            qvals[mask] = multipletests(pvals[mask], method='fdr_bh')[1]

        summary = pd.DataFrame({
            'immune_cell': immune_cols,
            f'mean_{g1}': means.loc[g1, immune_cols].values,
            f'mean_{g2}': means.loc[g2, immune_cols].values,
            'delta_mean': delta[immune_cols].values,
            'pval': pvals,
            'qval': qvals,
            f'n_{g1}': n1,
            f'n_{g2}': n2,
        })
        summary['direction'] = np.where(
            summary['delta_mean'] > 0, f'Up in {g2}',
            np.where(summary['delta_mean'] < 0, f'Up in {g1}', 'No difference'),
        )
        summary = summary.sort_values('delta_mean')

        print(f"N ({g1}) = {(merged[group_col] == g1).sum()} | N ({g2}) = {(merged[group_col] == g2).sum()}")
        return summary


    def p_to_stars(p):
        if pd.isna(p):
            return ''
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''


    def immune_barplot_summary(summary, title, cancer=True, star_source='pval',
                                save_dir=SAVE_DIR, figsize=(10, 8),
                                fontsize_y=20, fontsize_x=20, xlabel_fontsize=20):
        """Horizontal bar chart of mean immune-fraction difference per cell type,
        with significance stars, sorted from most up-in-group1 to up-in-group2."""
        plt.close('all')
        exclude_ = ['patient_id', 'Count_x', 'col', 'sample_id', 'Type', 'Type_y',
                    'cluster_type_x', 'cluster_type_y']
        summary = summary[~summary['immune_cell'].isin(exclude_)]
        order = rw_function.desired_order(cancer)
        g1, g2 = order[0], order[1]

        ordered = summary.sort_values('delta_mean').copy()

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(ordered))

        ax.barh(y, ordered['delta_mean'], color='black')
        ax.axvline(0, linewidth=2, linestyle='--', color='black')

        ax.set_yticks(y)
        ax.set_yticklabels(ordered['immune_cell'], fontsize=fontsize_y)
        ax.tick_params(axis='x', labelsize=fontsize_x)
        ax.set_xlabel(f"Mean fraction difference ({g2} - {g1})", fontsize=xlabel_fontsize)
        ax.set_title("")

        fig.canvas.draw()
        xmin, xmax = ax.get_xlim()
        dx = 0.01 * (xmax - xmin)

        for i, (xv, pv) in enumerate(zip(ordered['delta_mean'], ordered[star_source])):
            s = p_to_stars(pv)
            if not s:
                continue
            xtext = xv + (dx if xv >= 0 else -dx)
            ha = 'left' if xv >= 0 else 'right'
            ax.text(xtext, i, s, va='center', ha=ha, fontweight='bold', color='black', clip_on=False)

        plt.tight_layout()

        outdir = save_dir / 'immune'
        save_panel(outdir / f"immune_{g1}_vs_{g2}_{title}_barh_stars.png", dpi=300, bbox_inches="tight")
        plt.close('all')


    # -----------------------------------------------------------------------------
    # Pathway-class vs. immune-cell correlation heatmaps (side-by-side by group)
    # -----------------------------------------------------------------------------

    def stars_from(q):
        s = np.empty(q.shape, dtype=object)
        s[:] = ''
        s[(q < 0.05) & (q >= 0.01)] = '*'
        s[(q < 0.01) & (q >= 0.001)] = '**'
        s[q < 0.001] = '***'
        return pd.DataFrame(s, index=q.index, columns=q.columns)


    def corr_for_groups(df_1, df_2, cancer=True):
        """Pathway-class vs. immune-cell Spearman correlation, computed
        separately for each of the two comparison groups.

        df_1: per-sample glycogene z-score table (has group/Mixture/pathway
        class/zscore columns). df_2: CIBERSORTx immune-fraction table.
        """
        group_col = rw_function.col_type(cancer)
        group_names = rw_function.desired_order(cancer)

        group_col_1 = rw_function.find_col(df_1, group_col)
        mixture_col_1 = rw_function.find_col(df_1, 'Mixture')
        pathway_col_1 = rw_function.find_col(df_1, 'pathway class')
        zscore_col_1 = rw_function.find_col(df_1, 'zscore')
        mixture_col_2 = rw_function.find_col(df_2, 'Mixture')

        if group_col_1 is None:
            raise ValueError(f"Could not find group column '{group_col}' in df_1")
        if mixture_col_1 is None:
            raise ValueError("Could not find 'Mixture' in df_1")
        if pathway_col_1 is None:
            raise ValueError("Could not find 'pathway class' in df_1")
        if zscore_col_1 is None:
            raise ValueError("Could not find 'zscore' in df_1")
        if mixture_col_2 is None:
            raise ValueError("Could not find 'Mixture' in df_2")

        exclude_cols = {
            mixture_col_2,
            rw_function.find_col(df_2, 'patient_id'),
            rw_function.find_col(df_2, 'Type'),
            rw_function.find_col(df_2, 'cluster_type'),
            rw_function.find_col(df_2, 'Count'),
            rw_function.find_col(df_2, 'Count_x'),
            rw_function.find_col(df_2, 'Count_y'),
        }
        exclude_cols = {c for c in exclude_cols if c is not None}

        immune_cols = [
            c for c in df_2.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_2[c])
        ]
        if len(immune_cols) == 0:
            raise ValueError("No numeric immune columns found in df_2")

        results = {}
        for group_name in group_names:
            sub = df_1.loc[
                df_1[group_col_1].astype(str).str.lower() == str(group_name).lower(),
                [mixture_col_1, pathway_col_1, zscore_col_1]
            ].copy()
            if sub.empty:
                raise ValueError(f"No rows for group='{group_name}' in df_1[{group_col_1}]")

            path_by_sample = (
                sub.groupby([mixture_col_1, pathway_col_1])[zscore_col_1]
                .mean()
                .unstack(pathway_col_1)
            )

            merged = path_by_sample.merge(
                df_2.set_index(mixture_col_2)[immune_cols],
                left_index=True, right_index=True, how='inner',
            )
            if merged.empty:
                raise ValueError(f"No overlap between {group_name} Mixture values and df_2")

            pathway_cols = list(path_by_sample.columns)
            merged_num = merged[pathway_cols + immune_cols].apply(pd.to_numeric, errors='coerce')

            M = merged_num.to_numpy(dtype=float)
            rho_full, p_full = spearmanr(M, axis=0, nan_policy='omit')

            P, I = len(pathway_cols), len(immune_cols)
            rho = pd.DataFrame(rho_full[:P, P:P + I], index=pathway_cols, columns=immune_cols)
            pval = pd.DataFrame(p_full[:P, P:P + I], index=pathway_cols, columns=immune_cols)

            flat = pval.to_numpy().ravel()
            mask = ~np.isnan(flat)
            q = np.full_like(flat, np.nan, dtype=float)
            if mask.any():
                q[mask] = multipletests(flat[mask], method='fdr_bh')[1]
            qval = pd.DataFrame(q.reshape(pval.shape), index=pval.index, columns=pval.columns)

            results[group_name] = {'rho': rho, 'pval': pval, 'qval': qval, 'merged': merged_num}

        return results


    def plot_pathway_vs_immune_by_group(
        df_1, df_2, title, cancer=True, show_stars=True,
        remove_rows=None, remove_cols=None, row_order=None, col_order=None,
        figsize=None, cmap='bwr', vmin=None, vmax=None, fontsize=14,
        xtick_rotation=90, ytick_rotation=0, save_dir=SAVE_DIR, show_panel_title=True,
    ):
        """Side-by-side pathway-class x immune-cell correlation heatmaps, one
        panel per group, with FDR-based significance stars and a shared colorbar."""
        results = corr_for_groups(df_1, df_2, cancer=cancer)
        g1, g2 = rw_function.desired_order(cancer)

        rho_1, q_1 = results[g1]['rho'].copy(), results[g1]['qval'].copy()
        rho_2, q_2 = results[g2]['rho'].copy(), results[g2]['qval'].copy()

        rows = sorted(set(rho_1.index) & set(rho_2.index))
        cols = sorted(set(rho_1.columns) & set(rho_2.columns))
        rho_1, rho_2 = rho_1.loc[rows, cols], rho_2.loc[rows, cols]
        q_1, q_2 = q_1.loc[rows, cols], q_2.loc[rows, cols]

        if remove_rows is not None:
            keep_rows = [r for r in rho_1.index if r not in remove_rows]
            rho_1, rho_2 = rho_1.loc[keep_rows], rho_2.loc[keep_rows]
            q_1, q_2 = q_1.loc[keep_rows], q_2.loc[keep_rows]

        if remove_cols is not None:
            keep_cols = [c for c in rho_1.columns if c not in remove_cols]
            rho_1, rho_2 = rho_1[keep_cols], rho_2[keep_cols]
            q_1, q_2 = q_1[keep_cols], q_2[keep_cols]

        if row_order is not None:
            keep_rows = [r for r in row_order if r in rho_1.index]
            rho_1, rho_2 = rho_1.loc[keep_rows], rho_2.loc[keep_rows]
            q_1, q_2 = q_1.loc[keep_rows], q_2.loc[keep_rows]

        if col_order is not None:
            keep_cols = [c for c in col_order if c in rho_1.columns]
            rho_1, rho_2 = rho_1[keep_cols], rho_2[keep_cols]
            q_1, q_2 = q_1[keep_cols], q_2[keep_cols]

        stars_1, stars_2 = stars_from(q_1), stars_from(q_2)

        if vmax is None:
            vmax = float(pd.concat([rho_1.stack(), rho_2.stack()]).abs().max())
        if vmin is None:
            vmin = -vmax

        if figsize is None:
            fig_w = max(16, 0.9 * len(rho_1.columns) + 4)
            fig_h = max(14, 0.35 * len(rho_1.index) + 2)
            figsize = (fig_w, fig_h)

        y_font = 22
        cbar_font = 30
        display_map = {'I': 'Subtype I', 'II': 'Subtype II'} if cancer else {}

        plt.close('all')
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        for i, (ax, R, S, panel_title) in enumerate([
            (axes[0], rho_1, stars_1, display_map.get(g1, g1)),
            (axes[1], rho_2, stars_2, display_map.get(g2, g2)),
        ]):
            sns.heatmap(
                R, cmap=cmap, center=0, vmin=vmin, vmax=vmax,
                linewidths=0.3, linecolor='lightgrey',
                annot=S if show_stars else False, fmt='',
                annot_kws={'color': 'black', 'fontsize': fontsize * 0.75},
                cbar=False, ax=ax,
            )
            if show_panel_title:
                ax.set_title(panel_title, fontsize=fontsize + 2)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', labelsize=y_font)
            plt.setp(ax.get_xticklabels(), rotation=xtick_rotation, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=ytick_rotation)

        plt.tight_layout(rect=[0, 0, 0.9, 1])

        # Align the colorbar's bottom with the heatmap panels' bottom (not the
        # x-tick labels below them), placed just to the right of the figure.
        pos = axes[1].get_position()
        cbar_ax = fig.add_axes([0.92, pos.y0, (pos.height * 0.3) / 30, pos.height * 0.3])
        cbar = fig.colorbar(axes[1].collections[0], cax=cbar_ax, label="Spearman r")
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        cbar.ax.tick_params(labelsize=cbar_font)
        cbar.set_label("Spearman r", fontsize=cbar_font)

        outdir = save_dir / 'pathway_immune'
        save_panel(outdir / f"pathway_immune_ind_{title}.PNG", dpi=600, bbox_inches="tight")
        plt.close('all')


    def plot_difference(df_1, df_2, title, cancer=True, show_values=False, save_dir=SAVE_DIR):
        """Single heatmap of the pathway-immune correlation difference (group2 - group1)."""
        results = corr_for_groups(df_1, df_2, cancer=cancer)
        g1, g2 = rw_function.desired_order(cancer)

        rho_1 = results[g1]['rho']
        rho_2 = results[g2]['rho']

        rows = sorted(set(rho_1.index) & set(rho_2.index))
        cols = sorted(set(rho_1.columns) & set(rho_2.columns))
        delta_rho = rho_2.loc[rows, cols] - rho_1.loc[rows, cols]

        vmax = float(delta_rho.abs().max().max())
        y_font = 24

        plt.close('all')
        fig, ax = plt.subplots(figsize=(14, 1.2 * max(6, 0.5 * len(rows))))
        sns.heatmap(
            delta_rho, cmap='bwr', center=0, vmin=-vmax, vmax=vmax,
            annot=show_values, fmt='.2f' if show_values else '',
            linewidths=0.3, linecolor='lightgrey', ax=ax, cbar=False,
        )
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.setp(ax.get_xticklabels(), fontsize=y_font, rotation=90, ha='right')
        plt.setp(ax.get_yticklabels(), fontsize=y_font, rotation=0)

        fig.tight_layout()
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.02, pos.height * 0.3])
        cbar_label = f'$r_{{{g2}}} - r_{{{g1}}}$'
        cbar = fig.colorbar(ax.collections[0], cax=cbar_ax, label=cbar_label)
        cbar.ax.tick_params(labelsize=30)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        cbar.set_label(cbar_label, fontsize=30)

        outdir = save_dir / 'pathway_immune'
        save_panel(outdir / f"pathway_immune_diff_{title}.PNG", dpi=600, bbox_inches="tight")
        plt.close('all')


    # -----------------------------------------------------------------------------
    # Main workflow
    # -----------------------------------------------------------------------------

    def main():

        df_nca_immune, df_ca_immune = immune_nca, immune_ca
        df_glyco_path_ca, df_glyco_path_nca = gene_score_psample_ca, gene_score_psample_nca
        df_gly_path_ca = df_gly_path(df_glyco_path_ca)
        df_gly_path_nca = df_gly_path(df_glyco_path_nca)

        df_ca_immune_gly_sum = immune_group_summary(df_ca_immune, df_gly_path_ca, 'ca', cancer=True, save_dir=SAVE_DIR)
        df_nca_immune_gly_sum = immune_group_summary(df_nca_immune, df_gly_path_nca, 'nca', cancer=False, save_dir=SAVE_DIR)

        immune_barplot_summary(df_ca_immune_gly_sum, "ca_subtype", cancer=True, save_dir=SAVE_DIR)
        immune_barplot_summary(df_nca_immune_gly_sum, "nca_", cancer=False, save_dir=SAVE_DIR)

        plot_pathway_vs_immune_by_group(df_gly_path_ca, df_ca_immune, 'ca_subtype', cancer=True,
                                         save_dir=SAVE_DIR)
        plot_pathway_vs_immune_by_group(df_gly_path_nca, df_nca_immune, 'nca', cancer=False, save_dir=SAVE_DIR)

        plot_difference(df_gly_path_nca, df_nca_immune, 'nca', cancer=False, show_values=False, save_dir=SAVE_DIR)
        plot_difference(df_gly_path_ca, df_ca_immune, 'ca', cancer=True, show_values=False, save_dir=SAVE_DIR)

        print(f"{SCRIPT_NAME}: panels done")


    main()

# =============================================================================
# stage_05_ccrcc_nca_glycoprotein_go_enrichment
# =============================================================================

def stage_05_ccrcc_nca_glycoprotein_go_enrichment(deg_results):
    """GO term / pathway enrichment analysis for the Normal-vs-Cancer DEGs from
    stage_02, for two gene panels:
      - "glycoprotein": the glycan-based (glycoprotein) gene comparison
        (Figure1 B).
      - "global": the all-genes comparison (Figure1 A).

    Pipeline
    --------
    1. Start from each panel's FDR < 0.05 DEG table (stage_02, `deg_table`;
       Benjamini-Hochberg over all tested genes). An additional |log2 fold change| > 0.5
       filter is applied on top of that. For "global" specifically, the
       remaining genes are further narrowed to the top 1500 by p-value (across
       both directions combined) -- the FDR/log2FC-filtered set alone is still
       too broad and unspecific (5000+ genes) for GO overrepresentation
       testing to give a sharp signal.
    2. Split the remaining genes into upregulated / downregulated sets by the
       sign of the log2 fold change (Cancer vs. Normal).
    3. For each panel and direction, run Enrichr (`gseapy.enrichr`) against
       three gene-set libraries: GO Biological Process, GO Molecular Function
       and GO Cellular Component. Each query returns terms ranked by (Fisher's-exact-test)
       p-value, BH-FDR corrected ("Adjusted P-value").
    4. Plot the top 15 terms (by adjusted p-value) as a horizontal bar chart
       of -log10(adjusted p).
    5. Additionally, for each panel, run preranked GSEA (`gseapy.prerank`)
       against KEGG using *every* tested gene in that panel (the full
       ~4,000/~32,754-gene test set from stage_02) -- no
       significance/effect-size filtering, since GSEA's power comes from
       using the whole ranked list rather than a pre-selected subset. Genes
       are ranked by a signed statistic, -log10(p_uncorrected) x
       sign(log2 fold change) (the `t` column of the test table
       is always NaN -- `ab_test_plot` computes the t-statistic but does
       not store it -- so this signed p-value
       is used as the ranking metric instead of a moderated t-statistic).

    Requires network access (Enrichr is a hosted web service) and the `gseapy`
    package. Panels are rendered in memory.
    """
    _reset_mpl_defaults()

    import sys
    import textwrap
    from pathlib import Path

    import numpy as np
    import pandas as pd

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import gseapy as gp

    SCRIPT_NAME = "2026_ccrcc_nca_glycoprotein_go_enrichment"
    OUTPUT_DIR = RESULT_MANUSCRIPT_DIR / SCRIPT_NAME
    SAVE_DIR = OUTPUT_DIR

    # `deg_results` (from stage_02): {panel: (significant DEGs, every tested gene)}
    MIN_ABS_LOG2FC = 0.5
    GLOBAL_TOP_N = 1500

    GENE_SETS = {
        'GO_BP': 'GO_Biological_Process_2023',
        'GO_MF': 'GO_Molecular_Function_2023',
        'GO_CC': 'GO_Cellular_Component_2023',
    }

    GENE_SET_DISPLAY_NAMES = {
        'GO_BP': 'Biological Process',
        'GO_MF': 'Molecular Function',
        'GO_CC': 'Cellular Component',
        'KEGG': 'KEGG Pathway',
    }

    GSEA_GENE_SET = 'KEGG_2021_Human'
    GSEA_PERMUTATIONS = 1000
    GSEA_MIN_SIZE = 5
    GSEA_MAX_SIZE = 1000

    # Manuscript-figure title override for the GSEA bar chart (Figure6 A in
    # stage_06), keyed by panel_name.
    GSEA_TITLE_OVERRIDE = {
        'global': 'KEGG pathways from global genes ',
        'glycoprotein': 'KEGG pathways from glycan-based genes ',
    }

    # Manuscript-figure title override, keyed by (panel_name, label,
    # gene_set_key). Two different uses:
    #  - The two GO_MF panels that stand alone (SFigure7 A in
    #    stage_06) get the full "Molecular Function for ...
    #    genes" title baked in directly, since each is the only plot in its
    #    own panel (no composite-level row title above it).
    #  - The GO_BP/CC/MF panels sharing a row (Figure6 A/B, SFigure7 B/C)
    #    already get that gene-set/direction context from one shared
    #    composite-level row title instead, so these just need the full
    #    ontology name to distinguish the panels within that row.
    ENRICHMENT_TITLE_OVERRIDE = {
        ('global', 'upregulated', 'GO_MF'): 'Molecular Function for global upregulated genes ',
        ('glycoprotein', 'upregulated', 'GO_MF'): 'Molecular Function for glycan-based upregulated genes ',

        ('global', 'upregulated', 'GO_BP'): 'Biological Process',
        ('global', 'upregulated', 'GO_CC'): 'Cellular Component',
        ('glycoprotein', 'upregulated', 'GO_BP'): 'Biological Process',
        ('glycoprotein', 'upregulated', 'GO_CC'): 'Cellular Component',

        ('global', 'downregulated', 'GO_BP'): 'Biological Process',
        ('global', 'downregulated', 'GO_CC'): 'Cellular Component',
        ('global', 'downregulated', 'GO_MF'): 'Molecular Function',
        ('glycoprotein', 'downregulated', 'GO_BP'): 'Biological Process',
        ('glycoprotein', 'downregulated', 'GO_CC'): 'Cellular Component',
        ('glycoprotein', 'downregulated', 'GO_MF'): 'Molecular Function',
    }


    # -----------------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------------

    def load_degs_all_with_log2fc(df, min_abs_log2fc=MIN_ABS_LOG2FC):
        """FDR < 0.05 DEGs (Normal vs. Cancer, from stage_02's `deg_table`),
        additionally filtered to |log2 fold change| > `min_abs_log2fc` and
        sorted by p-value."""
        df = df.dropna(subset=['gene', 'log2_fold_change', 'regulation']).copy()
        df = df[df['log2_fold_change'].abs() > min_abs_log2fc]
        df = df.sort_values('p_uncorrected', ascending=True).reset_index(drop=True)
        return df


    def build_rank_metric(df):
        """Signed ranking metric for preranked GSEA, using *every* tested gene
        in the panel (no filtering): -log10(p_uncorrected) x sign(log2 fold
        change). See the module docstring for why this is used in place of
        the (unpopulated) `t` column.
        """
        df = df.dropna(subset=['gene', 'log2_fold_change', 'p_uncorrected']).copy()
        df = df.drop_duplicates(subset='gene', keep='first')

        p = df['p_uncorrected'].clip(lower=1e-300)
        df['rank_metric'] = -np.log10(p) * np.sign(df['log2_fold_change'])
        rnk = df[['gene', 'rank_metric']].sort_values('rank_metric', ascending=False).reset_index(drop=True)
        return rnk


    TERM_LABEL_WRAP_WIDTH = 35


    def wrap_term_label(text, width=TERM_LABEL_WRAP_WIDTH):
        """Wrap a term/pathway label onto at most 2 lines of at most `width`
        characters each (longer terms are ellipsized on the 2nd line), so no
        single label's length can stretch a panel's left margin wider than any
        other's -- keeping every bar-chart panel's aspect ratio (and so, once
        composited, its apparent fontsize) the same regardless of how long its
        longest label happens to be."""
        lines = textwrap.wrap(text, width=width, break_long_words=False)
        if len(lines) <= 1:
            return text
        if len(lines) == 2:
            return '\n'.join(lines)
        return f"{lines[0]}\n{lines[1]}..."


    # -----------------------------------------------------------------------------
    # Enrichment
    # -----------------------------------------------------------------------------

    def run_enrichment(genes, panel_name, label, gene_set_key, gene_set_name, save_dir, title_override=None,
                        height_scale=1.0):
        """Run one Enrichr query and plot the top enriched terms (by adjusted
        p-value), saved as `{panel_name}_{label}_{gene_set_key}_top_terms.png`."""
        genes = list(pd.unique(genes))
        if len(genes) == 0:
            print(f"{panel_name}/{label} ({gene_set_key}): no genes, skipping")
            return None

        # Enrichr rate-limits rapid queries (HTTP 429): retry with a growing wait.
        for attempt in range(6):
            try:
                res = gp.enrichr(gene_list=genes, gene_sets=[gene_set_name], organism='human',
                                  outdir=None, no_plot=True)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 30 * (attempt + 1)
                print(f"Enrichr request failed ({e}); retrying in {wait} s")
                time.sleep(wait)
        df_res = res.results.sort_values('Adjusted P-value', ascending=True).reset_index(drop=True)

        stem = f"{panel_name}_{label}_{gene_set_key}"

        n_sig = int((df_res['Adjusted P-value'] < 0.05).sum())
        print(f"{panel_name}/{label} ({gene_set_key}): {len(genes)} genes queried, "
              f"{n_sig} terms with adjusted p < 0.05")

        # Downregulated panels end up 3-to-a-row in the manuscript figure
        # (SFigure7 B/C) vs. 2-to-a-row for upregulated (Figure6 A/B, SFigure7
        # A) -- the narrower column compresses the same source fontsize
        # smaller, so bump it here to come out the same size once composited.
        # title_fontsize is scaled by the same ratio, for the same reason.
        ytick_fontsize = 28 if label == 'downregulated' else 25
        title_fontsize = 31 * ytick_fontsize / 25
        plot_top_terms(df_res, panel_name, label, gene_set_key, save_dir, stem,
                        show_title=(gene_set_key == 'KEGG'), title_override=title_override,
                        ytick_fontsize=ytick_fontsize, title_fontsize=title_fontsize,
                        height_scale=height_scale)
        return df_res


    def plot_top_terms(df_res, panel_name, label, gene_set_key, out_dir, stem, top_n=15,
                        show_title=True, title_override=None, ytick_fontsize=25, title_fontsize=31,
                        height_scale=1.0):
        """Horizontal bar chart of the top enriched terms, most significant at
        the top. `height_scale` scales the panel's height only (width is
        always 10in), for a panel that should read a little taller without
        changing its width once composited."""
        top = df_res.head(top_n).iloc[::-1].copy()
        if top.empty:
            return

        neglog_p = -np.log10(top['Adjusted P-value'].clip(lower=1e-300))
        terms = top['Term'].str.replace(r'\s*\(GO:\d+\)', '', regex=True).apply(wrap_term_label)

        # taller per row when the yticklabels are bigger (so they don't crowd/
        # overlap vertically) and to leave room for a wrapped 2-line label.
        row_height = 0.75 * (ytick_fontsize / 25) * height_scale
        fig, ax = plt.subplots(figsize=(10, max(4, row_height * len(top) + 1)))
        ax.barh(terms, neglog_p, color='black')
        ax.set_xlabel('$-\\log_{10}\\,p$', fontsize=25)
        ax.set_ylabel('')
        plt.setp(ax.get_yticklabels(), fontsize=ytick_fontsize)
        plt.setp(ax.get_xticklabels(), fontsize=ytick_fontsize)
        ax.axvline(-np.log10(0.05), color='0.5', linestyle='--', lw=1)
        if title_override is not None:
            # Either this file is a single-plot panel in the manuscript figure
            # (its title baked in directly, since there's no composite-level
            # row title above it), or it shares a row with other GO_BP/CC/MF
            # panels under one shared composite-level row title (stage_06)
            # and this is just its own ontology-name sub-title.
            ax.set_title(title_override, fontsize=title_fontsize, pad=12, fontweight='bold')
        elif show_title:
            display_name = GENE_SET_DISPLAY_NAMES.get(gene_set_key, gene_set_key)
            ax.set_title(f"{display_name} ({panel_name}, {label})", fontsize=18)

        save_panel(out_dir / f"{stem}_top_terms.png", bbox_inches='tight', dpi=300)
        plt.close()


    def run_gsea_prerank(rnk, panel_name, gene_set_name=GSEA_GENE_SET, save_dir=SAVE_DIR, title=None):
        """Preranked GSEA against one gene-set library, using every ranked
        gene (no filtering), and plot the top pathways (by FDR q-value) as a
        horizontal bar chart of NES."""
        res = gp.prerank(rnk=rnk, gene_sets=[gene_set_name], min_size=GSEA_MIN_SIZE,
                          max_size=GSEA_MAX_SIZE, permutation_num=GSEA_PERMUTATIONS,
                          outdir=None, no_plot=True, seed=123, threads=4)
        df_res = res.res2d.copy()
        df_res['NES'] = df_res['NES'].astype(float)
        df_res['NOM p-val'] = df_res['NOM p-val'].astype(float)
        df_res['FDR q-val'] = df_res['FDR q-val'].astype(float)
        df_res = df_res.sort_values('FDR q-val', ascending=True).reset_index(drop=True)


        n_sig = int((df_res['FDR q-val'] < 0.25).sum())
        print(f"{panel_name}/GSEA (KEGG): {len(rnk)} genes ranked, "
              f"{n_sig} pathways with FDR q < 0.25")

        plot_gsea_top_terms(df_res, panel_name, gene_set_name, save_dir, title=title)
        return df_res


    def plot_gsea_top_terms(df_res, panel_name, gene_set_name, out_dir, top_n=15, title=None):
        """Horizontal bar chart of the top pathways by FDR q-value, as
        normalized enrichment score (NES) -- positive means enriched among
        genes ranked toward the top (upregulated) of the list, negative
        toward the bottom (downregulated)."""
        top = df_res.head(top_n).iloc[::-1].copy()
        if top.empty:
            return

        terms = top['Term'].str.replace(f'{gene_set_name}__', '', regex=False).apply(wrap_term_label)

        # 0.75/row (not 0.45): matches plot_top_terms's per-row height, leaving
        # room for a wrapped 2-line label.
        fig, ax = plt.subplots(figsize=(10, max(4, 0.75 * len(top) + 1)))
        ax.barh(terms, top['NES'], color='black')
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel('Normalized Enrichment Score', fontsize=25)
        ax.set_ylabel('')
        plt.setp(ax.get_yticklabels(), fontsize=25)
        plt.setp(ax.get_xticklabels(), fontsize=25)
        ax.set_title(title or f"KEGG Pathway (GSEA, {panel_name})", fontsize=31, pad=14, fontweight='bold')

        save_panel(out_dir / f"{panel_name}_gsea_top_pathways.png", bbox_inches='tight', dpi=300)
        plt.close()


    # -----------------------------------------------------------------------------
    # Main workflow
    # -----------------------------------------------------------------------------

    def main():

        for panel_name in ('glycoprotein', 'global'):
            df_degs, df_full = deg_results[panel_name]
            df_sig = load_degs_all_with_log2fc(df_degs, MIN_ABS_LOG2FC)

            selection_desc = f"FDR < 0.05 and |log2FC| > {MIN_ABS_LOG2FC}"
            if panel_name == 'global':
                # already sorted by p_uncorrected ascending; take the top
                # GLOBAL_TOP_N by significance across both directions combined
                # -- the FDR/log2FC-filtered set alone (5503 up / 194 down) is
                # too broad and unspecific for GO overrepresentation testing
                # to give a sharp signal (see the upregulated-direction 0-hit
                # result without this restriction).
                df_sig = df_sig.head(GLOBAL_TOP_N).reset_index(drop=True)
                selection_desc += f", top {GLOBAL_TOP_N} by p-value"


            up_genes = df_sig.loc[df_sig['regulation'] == 'upregulated', 'gene']
            down_genes = df_sig.loc[df_sig['regulation'] == 'downregulated', 'gene']
            print(f"\n{panel_name} genes (Normal vs. Cancer), {selection_desc}: "
                  f"{len(up_genes)} upregulated, {len(down_genes)} downregulated")

            for gene_set_key, gene_set_name in GENE_SETS.items():
                # Figure6 B/C (upregulated GO_BP/GO_CC, 2-col rows) and
                # SFigure7 B/C (downregulated GO_BP/GO_CC/GO_MF, 3-col rows) are
                # a little taller than panel A above them, same width -- every
                # panel sharing one of those rows needs the same height_scale
                # (else they'd mismatch each other, not just row A). GO_MF
                # upregulated (SFigure7 A) shares its row only with its own
                # sibling panel, so it can scale independently -- taller still,
                # to stop its (fewer, longer) term labels from overlapping.
                up_height_scale = 1.3 if gene_set_key in ('GO_BP', 'GO_CC') else (1.6 if gene_set_key == 'GO_MF' else 1.0)
                down_height_scale = 1.15 if gene_set_key in ('GO_BP', 'GO_CC', 'GO_MF') else 1.0
                run_enrichment(up_genes, panel_name, 'upregulated', gene_set_key, gene_set_name, save_dir=SAVE_DIR,
                                title_override=ENRICHMENT_TITLE_OVERRIDE.get((panel_name, 'upregulated', gene_set_key)),
                                height_scale=up_height_scale)
                run_enrichment(down_genes, panel_name, 'downregulated', gene_set_key, gene_set_name, save_dir=SAVE_DIR,
                                title_override=ENRICHMENT_TITLE_OVERRIDE.get((panel_name, 'downregulated', gene_set_key)),
                                height_scale=down_height_scale)

            # Preranked GSEA, separate from the Enrichr overrepresentation
            # analysis above: uses every tested gene in the panel (no
            # significance/effect-size filtering).
            rnk = build_rank_metric(df_full)
            print(f"{panel_name}: ranked GSEA using all {len(rnk)} tested genes")
            run_gsea_prerank(rnk, panel_name, GSEA_GENE_SET, save_dir=SAVE_DIR,
                             title=GSEA_TITLE_OVERRIDE.get(panel_name))

        print(f"{SCRIPT_NAME}: panels done")


    main()

# =============================================================================
# stage_06_figure_plot
# =============================================================================

def stage_06_figure_plot():
    """Composite manuscript figures (Figure1-6, SFigure1, 2, 4-7), assembled
    from the panel PNGs drawn by stages 01-05 earlier in this run. Outputs go
    to `2026_figure_plot/` next to this file.

    Each figure is placed on a fixed 7.5-inch-wide canvas, no taller than 9.4
    inches, saved at 600 dpi. Source images are placed with `imshow`, which
    preserves each image's own pixel aspect ratio (no distortion) and letterboxes
    it within its grid cell if the cell doesn't match that aspect exactly.
    """
    _reset_mpl_defaults()

    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from PIL import Image

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # One panel PNG (SFigure7) is ~179M pixels; disable Pillow's
    # decompression-bomb guard for this legitimate scientific figure.
    Image.MAX_IMAGE_PIXELS = None

    OUTPUT_DIR = FIGURE_DIR
    SAVE_DIR = OUTPUT_DIR

    FIG_WIDTH = 7.5
    MAX_HEIGHT = 9.4
    DPI = 600

    # All row layouts below use 1, 2, 3 or 6 equal-width panels per row (each
    # evenly divides this), plus a few rows with weighted (non-equal) column
    # splits -- 60 gives those enough rounding precision to hit their intended
    # ratios instead of collapsing back to an even split.
    GRID_COLS = 60


    def P(rel_path):
        """Virtual path of a panel rendered earlier in this run (see `save_panel`)."""
        return RESULT_MANUSCRIPT_DIR / rel_path


    _IMAGE_CACHE = {}


    def load_cropped_image(path, pad_bottom=0.0, pad_left=0.0):
        """Load a panel PNG and trim its white/blank border (each panel was
        saved with its own margins, so panels otherwise don't align by content
        across a row). Returns (array, width/height).

        `pad_bottom`: extra white space added below the cropped content, as a
        fraction of the cropped height -- lets one panel's visible content be
        pulled up within its cell to align with a neighboring panel's content
        (e.g. a heatmap with no x-tick-label margin next to one that has one).
        `pad_left`: same idea but adds blank width on the left, as a fraction of
        the cropped width -- nudges the visible content right within its cell.
        """
        key = (str(path), pad_bottom, pad_left)
        if key not in _IMAGE_CACHE:
            with open_panel(path) as im:
                arr = np.array(im.convert('RGB'))
            mask = (arr < 250).any(axis=2)
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if len(rows) and len(cols):
                pad = 4
                r0, r1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad + 1, arr.shape[0])
                c0, c1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad + 1, arr.shape[1])
                arr = arr[r0:r1, c0:c1]
            if pad_bottom:
                extra = int(round(arr.shape[0] * pad_bottom))
                arr = np.vstack([arr, np.full((extra, arr.shape[1], 3), 255, dtype=arr.dtype)])
            if pad_left:
                extra = int(round(arr.shape[1] * pad_left))
                arr = np.hstack([np.full((arr.shape[0], extra, 3), 255, dtype=arr.dtype), arr])
            h, w = arr.shape[:2]
            _IMAGE_CACHE[key] = (arr, w / h)
        return _IMAGE_CACHE[key]


    def img_aspect(path, pad_bottom=0.0, pad_left=0.0):
        return load_cropped_image(path, pad_bottom, pad_left)[1]


    def col_spans(weights, total_cols=GRID_COLS):
        """Column (start, end) for each panel in a row, proportional to `weights`
        and covering the full `total_cols` width (cumulative rounding avoids gaps
        or overlap). weight=1 for every panel reproduces the old equal-width split."""
        total_w = sum(weights)
        bounds = [0]
        cum = 0
        for wt in weights:
            cum += wt
            bounds.append(round(total_cols * cum / total_w))
        return [(bounds[i], bounds[i + 1]) for i in range(len(weights))]


    def entry_weight(entry):
        if isinstance(entry, dict):
            return entry.get('weight', 1)
        return entry[3] if len(entry) > 3 else 1


    def entry_needed_height(entry, col_w):
        """Height this column needs at width `col_w` for no letterboxing --
        a stack's items add up (they share the column, one above the other)."""
        if isinstance(entry, dict):
            total = 0.0
            for sub in entry['stack']:
                pad_bottom = sub[2] if len(sub) > 2 else 0.0
                total += col_w / img_aspect(sub[1], pad_bottom)
            return total
        pad_bottom = entry[2] if len(entry) > 2 else 0.0
        pad_left = entry[4] if len(entry) > 4 else 0.0
        return col_w / img_aspect(entry[1], pad_bottom, pad_left)


    def trim_blank_rows(path, dpi):
        """Crop blank (white) pixel rows off the top and bottom of a saved PNG
        in place, keeping its full width. Returns the new height in inches."""
        with Image.open(path) as im:
            arr = np.array(im.convert('RGB'))
        rows = np.where((arr < 250).any(axis=2).any(axis=1))[0]
        if len(rows):
            arr = arr[rows[0]:rows[-1] + 1]
        Image.fromarray(arr).save(path, dpi=(dpi, dpi))
        return arr.shape[0] / dpi


    def compose_figure(rows, filename, row_titles=None, fig_width=FIG_WIDTH,
                        max_height=MAX_HEIGHT, dpi=DPI, save_dir=SAVE_DIR, fill_height=False,
                        hspace=0.25, wspace=0.12, title_fontsize=16, top=0.94, title_height=0.5,
                        trim_top_bottom=False):
        """Build and save one composite figure.

        `rows`: list of rows, each row a list of entries. A normal entry is
        (label, path[, pad_bottom[, weight[, pad_left]]]) -- `weight` (default 1)
        sets a panel's column share relative to others in its row, for a panel
        that should be wider/narrower than an equal split; `pad_left` nudges the
        visible content right within its cell (blank space added on the left).
        An entry can also be
        {'stack': [(label, path[, pad_bottom]), ...], 'weight': w} -- several
        panels stacked directly on top of each other (no row gap between them)
        sharing one column slot, e.g. two short panels next to one tall one.
        `row_titles`: optional {row_index: title_text} -- a centered title placed
        in its own slim row immediately above that image row (used for Figure6's
        GO-term groups). `title_height` sets that slim row's height (same units
        as the image rows' computed heights, default 0.5) -- lower it to tighten
        the gap a title row adds between the image row above it and its own
        image row below.
        `fill_height`: normally the natural (content-driven) height is only ever
        shrunk to fit `max_height`, never stretched -- pass True to scale up to
        exactly `max_height` too, when the natural layout comes in under it.
        `top`: gridspec top boundary (default 0.94), leaving headroom above the
        first row for its panel labels -- pass closer to 1.0 (no headroom
        needed) for a figure whose panels are all unlabeled (`label=None`),
        otherwise that headroom is just blank space above row 0.
        `trim_top_bottom`: after saving, crop away any blank (white) rows above
        the first and below the last visible content, so the figure has no
        top/bottom margin -- width stays fixed at `fig_width`, only the height
        shrinks.
        """
        row_titles = row_titles or {}

        expanded = []  # ('title', text) | ('images', row)
        for i, row in enumerate(rows):
            if i in row_titles:
                expanded.append(('title', row_titles[i]))
            expanded.append(('images', row))

        heights = []
        for kind, payload in expanded:
            if kind == 'title':
                heights.append(title_height)
            else:
                weights = [entry_weight(entry) for entry in payload]
                spans = col_spans(weights)
                needed = [entry_needed_height(entry, fig_width * (c1 - c0) / GRID_COLS)
                          for entry, (c0, c1) in zip(payload, spans)]
                heights.append(max(needed))

        # GridSpec's `hspace` is a fraction of the *average* row height, spent on
        # (nrows-1) gaps *in addition to* the row heights themselves -- so a
        # figure sized to just `sum(heights)` ends up having every row shrunk
        # below its intended size to make room for those gaps (most visible as
        # under-filled, letterboxed panels in a figure with few rows). Inflate
        # the target height by that same gap overhead so GridSpec's own division
        # hands each row back exactly the height it was given here.
        gap_overhead = hspace * (len(heights) - 1) * (sum(heights) / len(heights)) if len(heights) > 1 else 0.0
        natural_total = sum(heights) + gap_overhead
        scale = max_height / natural_total if fill_height else min(1.0, max_height / natural_total)
        fig_height = natural_total * scale

        fig = plt.figure(figsize=(fig_width, fig_height))
        # Default matplotlib subplot margins (rcParams `figure.subplot.*`) would
        # otherwise reserve a large blank border around the whole grid -- pin the
        # grid to fill essentially the entire fixed canvas instead.
        gs = fig.add_gridspec(nrows=len(expanded), ncols=GRID_COLS,
                               height_ratios=heights, hspace=hspace, wspace=wspace,
                               left=0.02, right=0.99, top=top, bottom=0.01)

        for r, (kind, payload) in enumerate(expanded):
            if kind == 'title':
                ax = fig.add_subplot(gs[r, :])
                ax.axis('off')
                ax.text(0.5, 0.1, payload, ha='center', va='bottom',
                        fontsize=title_fontsize, fontweight='bold', transform=ax.transAxes)
                continue

            def place_panel(ax, label, path, pad_bottom, pad_left=0.0, label_dy=0.0, anchor='NW'):
                ax.imshow(load_cropped_image(path, pad_bottom, pad_left)[0])
                ax.axis('off')
                # Anchor any aspect-driven letterboxing to the top-left corner
                # instead of centering, so panels sharing a column (e.g. A/C,
                # B/D stacked in different rows) stay flush left, and panels
                # sharing a row stay flush top. A panel can opt into a
                # different anchor (e.g. 'N' to center horizontally) when asked.
                ax.set_anchor(anchor)
                if label:
                    # Anchored at the axes' own top-left corner (not offset further
                    # left/up) so the glyph can't be pushed past the canvas edge
                    # for narrow, left-column panels. label_dy nudges it further
                    # up/down (axes-fraction) for a specific panel when asked.
                    ax.text(0.0, 1.0 + label_dy, label, transform=ax.transAxes, fontsize=12,
                            fontweight='bold', ha='left', va='bottom')

            row = payload
            weights = [entry_weight(entry) for entry in row]
            spans = col_spans(weights)
            for (c0, c1), entry in zip(spans, row):
                if isinstance(entry, dict):
                    stack = entry['stack']
                    col_w = fig_width * (c1 - c0) / GRID_COLS
                    sub_heights = [col_w / img_aspect(s[1], s[2] if len(s) > 2 else 0.0) for s in stack]
                    inner_gs = gs[r, c0:c1].subgridspec(len(stack), 1, height_ratios=sub_heights, hspace=0.15)
                    for i, sub in enumerate(stack):
                        label, path = sub[0], sub[1]
                        pad_bottom = sub[2] if len(sub) > 2 else 0.0
                        place_panel(fig.add_subplot(inner_gs[i, 0]), label, path, pad_bottom)
                else:
                    label, path = entry[0], entry[1]
                    pad_bottom = entry[2] if len(entry) > 2 else 0.0
                    pad_left = entry[4] if len(entry) > 4 else 0.0
                    label_dy = entry[5] if len(entry) > 5 else 0.0
                    anchor = entry[6] if len(entry) > 6 else 'NW'
                    place_panel(fig.add_subplot(gs[r, c0:c1]), label, path, pad_bottom, pad_left, label_dy, anchor)

        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / filename, dpi=dpi)
        plt.close(fig)
        if trim_top_bottom:
            fig_height = trim_blank_rows(save_dir / filename, dpi)
        print(f"Saved {filename} ({fig_width}in x {fig_height:.2f}in @ {dpi}dpi)")


    # -----------------------------------------------------------------------------
    # Figure 1
    # -----------------------------------------------------------------------------

    def build_figure1():
        rows = [
            [('A', P('2026_ccrcc_nca_volcano/nca_glob-log_volcano.png')),
             ('B', P('2026_ccrcc_nca_volcano/glycoproteins_nca_log_volcano.png')),
             ('C', P('2026_ccrcc_nca_volcano/glycogene_nca-log10_volcano.png'))],
            # weighted by aspect (D~1.16, E~1.20, F~1.31) so all three fill their
            # full cell height evenly, same technique as row3 below.
            [('D', P('2026_ccrcc_nca_volcano/nca_1500_heatmap.png'), 0.0, 1.16),
             ('E', P('2026_ccrcc_subtype/gscore/ccRCC_norm_ca_full_gene_score.png'), 0.0, 1.20),
             ('F', P('2026_ccrcc_subtype/PCA/ccRCC_norm_ca_full_fc_pca.png'), 0.0, 1.31)],
            # weighted by aspect (G~1.30, H~2.00) so both fill their full cell
            # height evenly. I (sialyltransferase) moved to Figure5.
            [('G', P('2026_ccrcc_subtype/PCA/ccRCC_norm_ca_full_nlogp_pca.png'), 0.0, 1.30),
             ('H', P('2026_ccrcc_subtype/gscore/ccRCC_norm_ca_top5pathways_genes.PNG'), 0.0, 2.00)],
            # J: the overview-plus-top5-pathways PCA strip (1 row x 6 panels,
            # very wide/short -- the dedicated one-row variant) gets its own
            # full-width row, its final row in this figure.
            [('J', P('2026_ccrcc_subtype/PCs/PCs_overview_top5_pathways_norm_ca_1row.png'))],
        ]
        # This figure's natural height (~10.47in) now exceeds the 9.4in default
        # max_height (added rows since that default was set) -- left uncapped,
        # every row's height alone gets scaled down to fit (width doesn't),
        # under-filling each row's assigned column widths with blank space
        # instead of every panel keeping its own true aspect ratio. See
        # Figure3's identical fix.
        return compose_figure(rows, 'Figure1.png', wspace=0.2, max_height=10.5, trim_top_bottom=True)


    # -----------------------------------------------------------------------------
    # Figure 2
    # -----------------------------------------------------------------------------

    def build_figure2():
        rows = [
            # weighted by aspect (A~0.94, B~1.27) so both fill their full cell
            # height evenly -- A's own taller aspect was otherwise driving the
            # row height at an equal 1:1 split, leaving B's shorter bar chart
            # letterboxed with blank space below it (read as a gap before row2).
            [('A', P('2026_ccrcc_subtype/immune/ccRCC-na-ca_immune_fraction_individual.PNG'), 0.0, 0.94),
             ('B', P('2026_cybertx_nca_immune_glyco_full/immune/immune_Normal_vs_Cancer_nca__barh_stars.png'), 0.0, 1.27)],
            [('C', P('2026_cybertx_nca_immune_glyco_full/pathway_immune/pathway_immune_diff_nca.PNG')),
             # C has x-axis tick labels below its heatmap, D doesn't -- pad D's
             # bottom so its heatmap (not just its bounding box) bottoms out at
             # roughly the same height as C's tick labels instead of lower.
             ('D', P('2026_ccrcc_subtype/cybersort_immune/ccRCC_nca_20-immune gene-genelist-heatmap.png'), 0.4, 1, 0.08)],
        ]
        return compose_figure(rows, 'Figure2.png')


    # -----------------------------------------------------------------------------
    # Figure 3
    # -----------------------------------------------------------------------------

    def build_figure3():
        d_path = P('2026_ccrcc_subtype/gscore/ccRCC_subtype_full_gene_score.png')
        e_path = P('2026_ccrcc_subtype/PCA/ccRCC_subtype_full_fc_pca.png')
        f_path = P('2026_ccrcc_subtype/PCA/ccRCC_subtype_full_nlogp_pca.png')
        g_path = P('2026_ccrcc_subtype/gscore/ccRCC_subtype_top5pathways_genes.PNG')
        i_path = P('2026_ccrcc_subtype/PCs/PCs_overview_top5_pathways_subtype.png')
        rows = [
            # A, B, C given equal column shares, same as D, E, F below -- keeps
            # both rows' column boundaries identical (A/D, B/E, C/F all align).
            [('A', P('2026_ccrcc_subtype/df_ccrcc_no_mito_all-cancer-only.PNG')),
             ('B', P('2026_ccrcc_subtype/survival/df_ccrcc_no_mito_all_survival.png')),
             # pad_left: GridSpec's wspace barely moves the boundary between B and
             # C at this GRID_COLS resolution, so force a real gap by padding
             # C's own image on the left instead.
             ('C', P('2026_ccrcc_subtype/subtype_by_glycogene_top10_pathway_heatmap.PNG'), 0.0, 1, 0.08)],
            # D, E, F share a row, weighted by each image's own aspect so all
            # three fill their full cell height evenly (read live via
            # img_aspect, so the weights follow regenerated source PNGs).
            [('D', d_path, 0.0, img_aspect(d_path)),
             ('E', e_path, 0.0, img_aspect(e_path)),
             ('F', f_path, 0.0, img_aspect(f_path))],
            # G (moved down from the D/E row) and I (the overview-plus-top5-
            # pathways PCA strip, 2 rows x 3 cols) share the last row, weighted
            # by aspect the same way.
            [('G', g_path, 0.0, img_aspect(g_path)),
             ('I', i_path, 0.0, img_aspect(i_path))],
        ]
        return compose_figure(rows, 'Figure3.png', wspace=0.2, trim_top_bottom=True)


    # -----------------------------------------------------------------------------
    # Figure 4
    # -----------------------------------------------------------------------------

    def build_figure4():
        rows = [
            # weighted by aspect (A~1.27, B~1.02) so both fill their full cell
            # height evenly -- A's own wider aspect was otherwise under-filling
            # its column at an equal 1:1 split (B, needing more height per unit
            # width, drove the row height), read as a gap before row 2.
            [('A', P('2026_ccrcc_subtype/immune/ccRCC-subtype_immune_fraction_individual.PNG'), 0.0, 1.27),
             ('B', P('2026_ccrcc_subtype/immune/ccRCC_na_ca_immune_cell_fraction_mean.PNG'), 0.0, 1.02)],
            [('C', P('2026_cybertx_nca_immune_glyco_full/immune/immune_I_vs_II_ca_subtype_barh_stars.png')),
             ('D', P('2026_cybertx_nca_immune_glyco_full/pathway_immune/pathway_immune_diff_ca.PNG'))],
        ]
        return compose_figure(rows, 'Figure4.png')


    # -----------------------------------------------------------------------------
    # Figure 5
    # -----------------------------------------------------------------------------

    def build_figure5():
        rows = [
            # A (aspect ~1.96, wide/short) and B (aspect ~0.99) share a row;
            # weighted by aspect so both fill their cell height evenly instead
            # of one being letterboxed short.
            [('A', P('2026_ccrcc_subtype/immune/ccRCC_subtype2_siglec_individual.PNG'), 0.0, 2),
             ('B', P('2026_ccrcc_subtype/cybersort_immune/ccRCC_subtype_20-immune gene-genelist-heatmap.png'), 0.0, 1)],
            # C: all 5 significant SIGLEC survival curves in one row (ncol=5).
            [('C', P('2026_ccRcc_survival/ca-only-Siglec-top_p-survival.PNG'))],
            # D (Figure3 H, aspect ~1.48) and E (Figure1 I, aspect ~2.28) share
            # a row; weighted by aspect so both fill their cell height evenly.
            [('D', P('2026_ccrcc_subtype/PCs/sialyltransferase_genes_subtype.PNG'), 0.0, 1.48),
             ('E', P('2026_ccrcc_subtype/PCs/sialyltransferase_genes_norm_ca.PNG'), 0.0, 2.28)],
        ]
        return compose_figure(rows, 'Figure5.png', wspace=0.01)


    # -----------------------------------------------------------------------------
    # Figure 6
    # -----------------------------------------------------------------------------

    def build_figure6():
        # Upregulated GO_BP/CC panels (global, then glycoprotein) and KEGG
        # GSEA; the downregulated and GO_MF panels are in SFigure7.
        go_dir = P('2026_ccrcc_nca_glycoprotein_go_enrichment')
        rows = [
            [('A', go_dir / 'global_upregulated_GO_BP_top_terms.png'),
             (None, go_dir / 'global_upregulated_GO_CC_top_terms.png')],
            [('B', go_dir / 'glycoprotein_upregulated_GO_BP_top_terms.png'),
             (None, go_dir / 'glycoprotein_upregulated_GO_CC_top_terms.png')],
            [('C', go_dir / 'global_gsea_top_pathways.png'),
             (None, go_dir / 'glycoprotein_gsea_top_pathways.png')],
        ]
        row_titles = {
            0: 'GO pathways for global upregulated genes',
            1: 'GO pathways for glycan-based upregulated genes',
        }
        # Natural height (~11.83in, B/C taller now) exceeds the 9.4in default
        # max_height -- left uncapped, every row's height alone gets scaled
        # down to fit (width doesn't), under-filling every column with blank
        # space instead of each panel keeping its own true aspect ratio. See
        # Figure1/Figure3's identical fix.
        # hspace/title_height reduced from the compose_figure defaults (0.25/0.5)
        # to shrink the blank gap the "GO pathways for glycan-based upregulated
        # genes" title row leaves between A's bottom and B's top -- scoped to
        # this figure only (SFigure7 keeps its own row_titles gaps unchanged).
        return compose_figure(rows, 'Figure6.png', row_titles=row_titles, hspace=0.04, wspace=0.25, title_fontsize=8,
                               title_height=0.3, max_height=11.9)


    # -----------------------------------------------------------------------------
    # Supplementary figures
    # -----------------------------------------------------------------------------

    def build_sfigure1():
        rows = [[(None, P('2026_ccrcc_subtype/PCs/PCs_top6_pathways_norm_ca_full_combined.PNG'))]]
        # No panel label on this single full-bleed image, so it doesn't need the
        # default top margin reserved for label headroom -- without `top` here,
        # that margin shows up as blank space above row 0.
        return compose_figure(rows, 'SFigure1.png', top=0.99)


    def build_sfigure2():
        rows = [
            [('A', P('2026_cybertx_nca_immune_glyco_full/pathway_immune/pathway_immune_ind_nca.PNG'))],
            # B (aspect ~0.65, narrow/tall) and C (aspect ~3.41, very wide/short)
            # weighted by aspect so both reach the same height -- this also
            # narrows B's column, since it needs far less width than C for that.
            [('B', P('2026_ccrcc_subtype/immune/ccRCC_na_ca_siglec_mean.PNG'), 0.0, 0.647, 0.0, 0.02),
             ('C', P('2026_ccrcc_subtype/immune/ccRCC_na_ca_siglec_individual.PNG'), 0.0, 3.407, 0.0, 0.02)],
        ]
        return compose_figure(rows, 'SFigure2.png')


    def build_sfigure4():
        rows = [[(None, P('2026_ccrcc_subtype/PCs/PCs_top6_pathways_subtype_full_combined.PNG'))]]
        # See build_sfigure1: no panel label here either, so skip the default
        # label-headroom margin instead of leaving it as blank space.
        return compose_figure(rows, 'SFigure4.png', top=0.99)


    def build_sfigure5():
        rows = [
            # A is much taller/narrower than B (aspect ~0.85 vs ~1.71) -- give B
            # roughly double A's column width so both fill their own cell height
            # exactly (no letterboxing), instead of one looking shorter.
            [('A', P('2026_ccrcc_subtype/immune/ccRCC_subtype_immune_cell_fraction_mean.PNG'), 0.0, 1),
             ('B', P('2026_cybertx_nca_immune_glyco_full/pathway_immune/pathway_immune_ind_ca_subtype.PNG'), 0.0, 2)],
        ]
        return compose_figure(rows, 'SFigure5.png')


    def build_sfigure6():
        rows = [[(None, P('2026_ccRcc_survival/ca-only-Siglec-all-survival.PNG'))]]
        return compose_figure(rows, 'SFigure6.png')


    def build_sfigure7():
        # A: the two upregulated GO_MF panels, sharing one row -- only the
        # first is labeled; their titles are drawn in the images themselves.
        # B/C: GO terms of the downregulated genes (global, glycoprotein).
        go_dir = P('2026_ccrcc_nca_glycoprotein_go_enrichment')
        rows = [
            [('A', go_dir / 'global_upregulated_GO_MF_top_terms.png'),
             (None, go_dir / 'glycoprotein_upregulated_GO_MF_top_terms.png')],
            [('B', go_dir / 'global_downregulated_GO_BP_top_terms.png'),
             (None, go_dir / 'global_downregulated_GO_CC_top_terms.png'),
             (None, go_dir / 'global_downregulated_GO_MF_top_terms.png')],
            [('C', go_dir / 'glycoprotein_downregulated_GO_BP_top_terms.png'),
             (None, go_dir / 'glycoprotein_downregulated_GO_CC_top_terms.png'),
             (None, go_dir / 'glycoprotein_downregulated_GO_MF_top_terms.png')],
        ]
        row_titles = {
            1: 'GO pathways for global downregulated genes',
            2: 'GO pathways for glycan-based downregulated genes',
        }
        return compose_figure(rows, 'SFigure7.png', row_titles=row_titles, hspace=0.08, wspace=0.25, title_fontsize=8)


    def main():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for build in [build_figure1, build_figure2, build_figure3, build_figure4,
                      build_figure5, build_figure6, build_sfigure1,
                      build_sfigure2, build_sfigure4, build_sfigure5, build_sfigure6,
                      build_sfigure7]:
            build()

        print(f"\nAll figures saved to: {OUTPUT_DIR.resolve()}")


    main()

def main():
    d1 = stage_01_ccrcc_subtype()
    deg_results = stage_02_ccrcc_nca_volcano(d1['df_nca'], d1['glycoprotein_genes'])
    stage_03_ccrcc_survival(d1['df_ca'])
    stage_04_cybertx_nca_immune_glyco_full(d1['immune_ca'], d1['immune_nca'],
                                           d1['gene_score_psample_ca'], d1['gene_score_psample_nca'])
    stage_05_ccrcc_nca_glycoprotein_go_enrichment(deg_results)
    stage_06_figure_plot()
    print(f"\nDone. Manuscript figures: {FIGURE_DIR}")


if __name__ == '__main__':
    main()
