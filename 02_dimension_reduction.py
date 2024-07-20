# %% md
# # 2. Dimension reduction and visualization of the TCGA-BRCA dataset


# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

pio.templates.default = 'simple_white'


def save_fig(fig_, savepath):
    fig_.update_layout(dragmode='pan', margin=dict(l=0, r=0, t=30, b=30))
    fig_.write_html(savepath, config={'scrollZoom': True, 'displaylogo': False})


# %%
dataset_full = pd.read_csv('data/processed/filtered_dataset.csv', index_col=0)
metadata_full = pd.read_csv('data/processed/metadata.csv', index_col=0)

metadata = metadata_full.dropna(subset='PAM50').reset_index(drop=True)
dataset = dataset_full.loc[:, metadata['submitter_id.samples']]

# %% md
# ## PCA

# %%
features_scaled = StandardScaler().fit_transform(dataset.T)

pca = PCA()
pca_model = pca.fit_transform(features_scaled)

cluster_values = pd.DataFrame(pca_model, columns=[f'PC{i + 1}' for i in range(pca_model.shape[1])])
cluster_values = cluster_values.join(metadata.rename(
    columns={'submitter_id.samples': 'SampleID',
             'sample_type.samples': 'SampleType'}))


# %%
fig = px.scatter(cluster_values, x='PC1', y='PC2', color='PAM50',
                 hover_data='SampleID',
                 title='PCA of TCGA-BRCA Dataset (all genes)')

output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)
fig.show()

fig.update_layout(
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.2,
        xanchor='center',
        x=0.5
    ),
    title='all genes',
    # showlegend=False
)

fig.update_layout(dragmode='pan', margin=dict(l=0, r=0, t=30, b=30))
fig.write_html(output_dir / 'pca_full_dataset.html',
               config={'scrollZoom': True,
                       'displaylogo': False})

# %%
# PCA Scree plot
explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

fig = px.line(x=range(1, len(cumulative_explained_variance) + 1),
              y=cumulative_explained_variance, markers=True)
fig.update_layout(
    title='PCA Elbow Plot',
    xaxis_title='Number of Components',
    yaxis_title='Cumulative Explained Variance',
    showlegend=False
)
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)
fig.show()

fig.update_layout(title=None)
save_fig(fig, output_dir / 'pca_full_scree.html')

# %% md
# As we can see, PCA of 2 components explains only 16% of data variance

# %%
# define functions for dimension reduction and plotting


def reduce_dimensions(method, dataset_, metadata_, method_name,
                      add_pca=False, n_pca_components=None, **kwargs):
    features_scaled_ = StandardScaler().fit_transform(dataset_.T)

    dr = method(**kwargs)  # `dr` as in Dimension Reduction
    dr_model = dr.fit_transform(features_scaled_)

    if add_pca:
        pca_ = PCA(n_components=n_pca_components)
        features_scaled_ = pca_.fit_transform(features_scaled_)

    cluster_values_ = pd.DataFrame(
        dr_model,
        columns=[f'{method_name}{i + 1}' for i in range(dr_model.shape[1])])
    cluster_values_ = cluster_values_.join(metadata_.rename(
        columns={'submitter_id.samples': 'SampleID',
                 'sample_type.samples': 'SampleType'}))
    return cluster_values_


def plot_cluster_values(cluster_values_, method_name, dataset_type, color='PAM50', **kwargs):
    fig_ = px.scatter(cluster_values_, x=f'{method_name}1', y=f'{method_name}2',
                      color=color, hover_data='SampleID',
                      title=f'{method_name} of TCGA-BRCA Dataset ({dataset_type})',
                      **kwargs)
    fig_.update_layout(
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
    )
    return fig_

# %% md
# We can also clusterize this dataset by SampleType:


# %%
cluster_values = reduce_dimensions(PCA, dataset_full, metadata_full, 'PCA')
fig = plot_cluster_values(cluster_values, 'PCA', 'all genes',
                          color='SampleType',
                          color_discrete_map={'Primary Tumor': px.colors.qualitative.D3[0],
                                              'Metastatic': px.colors.qualitative.D3[3],
                                              'Solid Tissue Normal': px.colors.qualitative.D3[1]})
fig.show()

fig.update_layout(
    title=None,
    xaxis_title='PC1',
    yaxis_title='PC2')
save_fig(fig, output_dir / 'pca_sample_type.html')

# %%
# create top 20% variance subset
gene_variances = dataset.var(axis=1)
top_20p_genes = gene_variances.nlargest(round(0.2 * len(dataset))).index
top20_var_subset = dataset.loc[top_20p_genes, :]


# %%
# create subset from DEGs
subtypes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
logFC = 1.5
pval = 0.05

subtype = subtypes[0]

degs = []
for subtype in subtypes:
    degs_df = pd.read_csv(f'results/diff_expression/{subtype}_vs_other.csv')
    # using only comparisons of each subtype vs others as we try
    # to identify different subtypes (and not comparing with healthy)
    degs.extend(degs_df[
        (degs_df['adj.P.Val'] < pval) & (degs_df['logFC'].abs() > logFC)
    ]['ID'].tolist())
degs = list(set(degs))
degs_subset = dataset.reindex(degs)
degs_subset.to_csv('data/processed/degs_subset.csv')


# %%
cluster_values = reduce_dimensions(PCA, dataset, metadata, 'PCA')
fig = plot_cluster_values(cluster_values, 'PCA', 'top 20% variable genes')
fig.show()
fig.update_layout(
    title='all genes',
    xaxis_title='PC1',
    yaxis_title='PC2')
save_fig(fig, output_dir / 'pca_full_dataset.html')

# %%
cluster_values = reduce_dimensions(PCA, top20_var_subset, metadata, 'PCA')
fig = plot_cluster_values(cluster_values, 'PCA', 'top 20% variable genes')
fig.show()
fig.update_layout(
    title='top 20% variable genes',
    xaxis_title='PC1',
    yaxis_title='PC2')
save_fig(fig, output_dir / 'pca_top20_var.html')

# %%
cluster_values = reduce_dimensions(PCA, degs_subset, metadata, 'PCA')
fig = plot_cluster_values(cluster_values, 'PCA', 'DEGs')
fig.show()
fig.update_layout(
    title='DEGs',
    xaxis_title='PC1',
    yaxis_title='PC2')

save_fig(fig, output_dir / 'pca_degs.html')


# %% md
# ## t-SNE

# %%
cluster_values = reduce_dimensions(TSNE, dataset, metadata, 'TSNE', perplexity=50)
fig = plot_cluster_values(cluster_values, 'TSNE', 'all genes')
fig.show()
fig.update_layout(
    title='all genes')
save_fig(fig, output_dir / 'tsne_full_dataset.html')

# %%
cluster_values = reduce_dimensions(TSNE, top20_var_subset, metadata, 'TSNE', perplexity=50)
fig = plot_cluster_values(cluster_values, 'TSNE', 'top 20% variable genes')
fig.show()
fig.update_layout(
    title='top 20% variable genes',
)
save_fig(fig, output_dir / 'tsne_top20_var.html')

# %%
cluster_values = reduce_dimensions(TSNE, degs_subset, metadata, 'TSNE', perplexity=50)
fig = plot_cluster_values(cluster_values, 'TSNE', 'DEGs')
fig.show()
fig.update_layout(
    title='DEGs')
save_fig(fig, output_dir / 'tsne_degs.html')


# %% md
# ## PCA + t-SNE


# %%
cluster_values = reduce_dimensions(TSNE, dataset, metadata, 'PCA+TSNE',
                                   add_pca=True, n_pca_components=346, perplexity=50)
fig = plot_cluster_values(cluster_values, 'PCA+TSNE', 'all genes')
fig.show()

fig.update_layout(
    title='all genes',
    xaxis_title='TSNE1',
    yaxis_title='TSNE2'
)
save_fig(fig, output_dir / 'pca_tsne_full_dataset.html')

# %%
cluster_values = reduce_dimensions(TSNE, top20_var_subset, metadata, 'PCA+TSNE',
                                   add_pca=True, n_pca_components=513, perplexity=50)
fig = plot_cluster_values(cluster_values, 'PCA+TSNE', 'top 20% variable genes')
fig.show()
fig.update_layout(
    title='top 20% variable genes',
    xaxis_title='TSNE1',
    yaxis_title='TSNE2'
)
save_fig(fig, output_dir / 'pca_tsne_top20_var.html')

# %%
cluster_values = reduce_dimensions(TSNE, degs_subset, metadata, 'PCA+TSNE',
                                   add_pca=True, n_pca_components=403, perplexity=50)
fig = plot_cluster_values(cluster_values, 'PCA+TSNE', 'DEGs')
fig.show()
fig.update_layout(
    title='DEGs',
    xaxis_title='TSNE1',
    yaxis_title='TSNE2'
)
save_fig(fig, output_dir / 'pca_tsne_degs.html')


# %% md
# ## UMAP

# %%
cluster_values = reduce_dimensions(UMAP, dataset, metadata, 'UMAP', n_neighbors=500)
fig = plot_cluster_values(cluster_values, 'UMAP', 'all genes')
fig.show()
fig.update_layout(
    title='all genes')
save_fig(fig, output_dir / 'umap_full_dataset.html')

# %%
cluster_values = reduce_dimensions(UMAP, top20_var_subset, metadata, 'UMAP', n_neighbors=500)
fig = plot_cluster_values(cluster_values, 'UMAP', 'top 20% variable genes')
fig.show()
fig.update_layout(
    title='top 20% variable genes')
save_fig(fig, output_dir / 'umap_top20_var.html')

# %%
cluster_values = reduce_dimensions(UMAP, degs_subset, metadata, 'UMAP', n_neighbors=500)
fig = plot_cluster_values(cluster_values, 'UMAP', 'DEGs')
fig.show()
fig.update_layout(
    title='DEGs')
save_fig(fig, output_dir / 'umap_degs.html')
