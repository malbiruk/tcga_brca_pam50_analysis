# %% md
# # 0. Preprocessing

# %%
from pathlib import Path

import pandas as pd
import plotly.figure_factory as ff
import plotly.io as pio

pio.templates.default = 'simple_white'

# %%
# load tables
input_dir = Path('data/input')
tcga = pd.read_csv(input_dir / 'TCGA-BRCA.htseq_fpkm-uq.tsv', sep='\t')
gene_names = pd.read_csv(input_dir / 'gencode.v22.annotation.gene.probeMap', sep='\t')
phenotypes = pd.read_csv(input_dir / 'TCGA-BRCA.GDC_phenotype.tsv', sep='\t')
lehmann_metadata = pd.read_csv(input_dir / 'Lehmann_metadata.csv')

# %%
# convert Enesmbl Ids to gene names
gene_names.rename(columns={'id': 'Ensembl_ID'}, inplace=True)
tcga = tcga.merge(gene_names[['Ensembl_ID', 'gene']], on='Ensembl_ID')
columns = ['Ensembl_ID', 'gene'] + [col for col in tcga.columns
                                    if col not in ['Ensembl_ID', 'gene']]
tcga = tcga[columns]
tcga.drop('Ensembl_ID', axis=1, inplace=True)
tcga = tcga.groupby('gene').sum()


# %%
print('N samples in `tcga`: ', len(tcga.columns))
print('N samples in `phenotypes`: ', phenotypes['submitter_id.samples'].nunique())
print('N samples in `lehmann_metadata`: ', lehmann_metadata['TCGA_SAMPLE'].nunique())

# %%
# save only common samples between `phenotypes` and `tcga`
phenodata = phenotypes[phenotypes['submitter_id.samples'].isin(tcga.columns)][
    ['submitter_id.samples', 'sample_type.samples']]

lehmann_metadata = lehmann_metadata.sort_values(by='TCGA_SAMPLE')
phenodata = phenodata.sort_values(by='submitter_id.samples')
tcga = tcga.reindex(sorted(tcga.columns), axis=1)

# %%
# create table with only tissue type and PAM50
phenodata['TCGA_SAMPLE'] = phenodata['submitter_id.samples'].str[:-1]
metadata = phenodata.merge(lehmann_metadata[['TCGA_SAMPLE', 'PAM50']],
                           on='TCGA_SAMPLE', how='left').reset_index(drop=True)

# %%
# using df.sample() as plotting too many values is resource heavy
tcga_sample = tcga.sample(6000, random_state=42)
genes_expr = tcga_sample.mean(axis=1)

fig = ff.create_distplot([genes_expr], group_labels=['Expression'])
fig.update_layout(
    xaxis_title='Expression',
    yaxis_title='Density',
    showlegend=False
)
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)
fig.show()
# fig.write_html(output_dir / 'average_expression_distribution.html')


# %% md
# As we can see, there is a huge density peak in low expression area (<7),
# and the density reduces a lot after about 28. Let's filter our dataset by this values.

# %%
genes_expr = tcga.mean(axis=1)
tcga_filtered = tcga[(genes_expr > 7) & (genes_expr < 28)]

# %%
# using sample as plotting too many values is resource heavy
tcga_sample = tcga_filtered.sample(6000, random_state=42)
genes_expr = tcga_sample.mean(axis=1)

fig = ff.create_distplot([genes_expr], group_labels=['Expression'])
fig.update_layout(title='Gene Expression Values Aggregated by Genes (after filtering)',
                  xaxis_title='Expression',
                  yaxis_title='Density',
                  )
fig.show()
# fig.write_html(output_dir / 'average_expression_distribution_filtered.html')


# %%
# saving filtered dataset and metadata
output_dir = Path('data/processed')
output_dir.mkdir(parents=True, exist_ok=True)
tcga_filtered.to_csv(output_dir / 'filtered_dataset.csv')
metadata.to_csv(output_dir / 'metadata.csv')
