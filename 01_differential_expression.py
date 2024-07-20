# %% markdown
# # 1. Differential Expression (DE) analysis


# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.graph_objs._figure import Figure
from scripts.limma import run_limma
from tqdm import tqdm

pio.templates.default = 'simple_white'


def save_fig(fig_, savepath):
    fig_.update_layout(dragmode='pan', margin=dict(l=30, r=30, t=30, b=30))
    fig_.write_html(savepath, config={'scrollZoom': True, 'displaylogo': False})


# %%
dataset = pd.read_csv('data/processed/filtered_dataset.csv', index_col=0)
metadata = pd.read_csv('data/processed/metadata.csv', index_col=0)


# %% md
# I am more comfortable using Python than R, so I'll use limma via this
# [script](https://github.com/shivaprasad-patil/LIMMA-Python-implementation).
# (I modified it a little: fixed some errors related to version differences,
# made final export to csv via pandas instead of xlsx via R, and
# wrapped everything into a function with arguments to be able to run it
# more easily in this notebook.)
#
# For each PAM50 Subtype N (N ∈ \[LumA, LumB, Her2, Basal, Normal\]) two type of the experiments
# will be carried:
# - Subtype N vs Normal samples
# - Subtype N vs other subtypes combined

# %%
# create experiment input files for scripts.limma.py for each of experiments
experiments = [{'pam50': pam50, 'experiment_type': exp_type}
               for pam50 in metadata['PAM50'].dropna().unique()
               for exp_type in ['vs_other', 'vs_normal']]

for experiment in experiments:
    if experiment['experiment_type'] == 'vs_other':
        sub_metadata = metadata.dropna(subset='PAM50')
        data = dataset.loc[:, sub_metadata['submitter_id.samples']]
        design = sub_metadata[['submitter_id.samples', 'PAM50']].rename(
            columns={'submitter_id.samples': 'ID', 'PAM50': 'Target'}
        ).reset_index(drop=True)
        design['Target'] = design['Target'].apply(
            lambda x: x if x == experiment['pam50'] else 'other')
        experiment['data'] = data
        experiment['design'] = design

    else:
        sub_metadata = metadata[
            (metadata['sample_type.samples'] != 'Metastatic')
            & ((metadata['PAM50'] == experiment['pam50']) | metadata['PAM50'].isna())]
        data = dataset.loc[:, sub_metadata['submitter_id.samples']]
        design = sub_metadata[['submitter_id.samples', 'sample_type.samples']].rename(
            columns={'submitter_id.samples': 'ID', 'sample_type.samples': 'Target'}
        ).reset_index(drop=True)
        design['Target'] = design['Target'].apply(
            lambda x: 'normal' if x == 'Solid Tissue Normal' else experiment['pam50'])
        experiment['data'] = data
        experiment['design'] = design


# %% md
# Now we have a list of dicts `experiments` where each element stores descriptions of each
# experiment as dict and contains the following keys and values:
# - `pam50`: PAM50 subtype
# - `experiment_type`: "vs_other" or "vs_normal"
# - `data`: pd.DataFrame with expression data only from relevant samples
# - `design`: design table which is required to run limma
# (describes which samples correspond to which groups)
#
# For example, let's display contents of the second element of this list:


# %%
display(experiments[1]['pam50'])
display(experiments[1]['experiment_type'])
display(experiments[1]['data'])
display(experiments[1]['design'])


# %%
# run limma for each experiment:
output_dir = Path('results/diff_expression')
output_dir.mkdir(parents=True, exist_ok=True)

for experiment in tqdm(experiments):
    run_limma(experiment['data'],
              experiment['design'],
              experiment['pam50'],
              experiment['experiment_type'].split('_', 1)[1],
              output_dir / f"{experiment['pam50']}_{experiment['experiment_type']}.csv")


# %% md
# some functions to get statistics and visualize DEGs are defined below

# %%
def calculate_n_degs(df: pd.DataFrame, logFC: float = 0.3, pval: float = 0.05) -> int:
    '''
    df:        table of DEGs (from limma)
    logFC:     threshold for DEGs by absolute value logFC
    pval:      threshold for DEGs by adjusted p-value

    returns n of DEGs
    '''
    return len(df[(abs(df['logFC']) > logFC) & (df['adj.P.Val'] < pval)])


# %%
def process_df_for_volcano(df: pd.DataFrame, logFC: float = 0.3, pval: float = 0.05,
                           remove_outliers: bool = True) -> pd.DataFrame:
    '''
    rename columns, add columns '-log10 P-value' and 'DEG type',
    remove outliers using interquantile range method

    df:                table of DEGs from limma
    logFC:             threshold for DEGs by absolute value logFC
    pval:              threshold for DEGs by adjusted p-value
    remove_outliers:   flag to remove genes with logFC value >= quartile + 15 * interquartile range
    '''
    df = df.copy()
    if remove_outliers:
        Q1 = df['logFC'].quantile(0.25)
        Q3 = df['logFC'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 15 * IQR
        upper_bound = Q3 + 15 * IQR
        were_genes = len(df)
        df = df[(df['logFC'] >= lower_bound) & (df['logFC'] <= upper_bound)]
        n_removed_genes = were_genes - len(df)
        if n_removed_genes != 0:
            print(f'{n_removed_genes} genes were removed as outliers')

    df['DEG type'] = df.apply(
        lambda x: 'Down-regulated genes'
        if ((x['logFC'] < -logFC) and (x['adj.P.Val'] < pval))
        else 'Up-regulated genes' if ((x['logFC'] > logFC) and (x['adj.P.Val'] < pval))
        else 'Insignificant genes',
        axis=1
    )
    df['-log10 P-value'] = -np.log10(df['adj.P.Val'])
    df.rename(columns={'logFC': 'log Fold Change'}, inplace=True)
    return df

# %%


def plot_degs(df: pd.DataFrame, logFC: float = 0.3, pval: float = 0.05,
              title: str = None) -> Figure:
    '''
    volcano plot

    df:        processed table of DEGs from limma (should include columns:
                   'ID', 'log Fold Change', '-log10 P-value', 'DEG type')
    logFC:     threshold for DEGs by absolute value logFC
    pval:      threshold for DEGs by adjusted p-value
    title:     title for plot
    '''
    fig = px.scatter(df, x='log Fold Change', y='-log10 P-value', color='DEG type',
                     hover_data=['ID'], title=title,
                     color_discrete_map={'Up-regulated genes': 'seagreen',
                                         'Down-regulated genes': 'pink',
                                         'Insignificant genes': 'darkgray'})
    fig.update_yaxes(range=[0, df['-log10 P-value'].max()])

    fig.add_shape(
        type='line',
        x0=logFC, x1=logFC,
        y0=0, y1=1,
        xref='x',
        yref='paper',
        line=dict(color='dimgray', width=0.5)
    )
    fig.add_shape(
        type='line',
        x0=-logFC, x1=-logFC,
        y0=0, y1=1,
        xref='x',
        yref='paper',
        line=dict(color='dimgray', width=0.5)
    )
    fig.add_shape(
        type='line',
        x0=0, x1=1,
        y0=-np.log10(pval), y1=-np.log10(pval),
        xref='paper',
        yref='y',
        line=dict(color='dimgray', width=0.5)
    )

    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
    )
    return fig


# %%
def describe_experiment(pam50: str, exp_type: str,
                        logFC: float = 0.3, pval: float = 0.05,
                        degs_dir: Path = Path('results/diff_expression'),
                        figs_dir: Path = Path('results/figures'),
                        save_figure: bool = True):
    degs = pd.read_csv(degs_dir / f'{pam50}_{exp_type}.csv')
    n_degs = calculate_n_degs(degs, logFC, pval)
    n_genes = len(degs)
    degs_percent = n_degs / n_genes * 100

    degs = process_df_for_volcano(degs, logFC, pval)

    fig = plot_degs(degs, logFC, pval, title=' '.join(f'{pam50}_{exp_type}'.split('_')))
    if save_figure:
        save_fig(fig, figs_dir / f'{pam50}_{exp_type}_degs.html')
    fig.show()

    print(f'\nN DEGs: {n_degs} ({round(degs_percent, 2)}%)')
    print('\nTop up-regulated genes:')
    display(degs[degs['DEG type'] == 'Up-regulated genes']
            .sort_values('log Fold Change', ascending=False)
            [['ID', 'log Fold Change', 'adj.P.Val']]
            .head())
    print('\nTop down-regulated genes:')
    display(degs[degs['DEG type'] == 'Down-regulated genes']
            .sort_values('log Fold Change')[['ID', 'log Fold Change', 'adj.P.Val']]
            .head())
    print('\n\n')


# %%
log_fc = 1.5
p_val = 0.05

# %%
for experiment in experiments:
    describe_experiment(experiment['pam50'], experiment['experiment_type'], log_fc, p_val,
                        save_figure=True)
