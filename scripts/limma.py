# -*- coding: utf-8 -*-
"""
@author: Shivaprasad
email:shivaprasad309319@gmail.com
"""

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


def run_limma(
    data: pd.DataFrame,
    design: pd.DataFrame,
    case: str,
    control: str,
    outfile: str,
) -> None:
    '''
    data:       Expression data in a matrix, where each column represents an experiment or
                sample ID, and each row represents a gene or probe expression
                (index should correspond to ID column in design)
    design:     Desgin matrix, where one column represents sample IDs, other (Target) --
                status of the data (normal, cancer, etc.)
    case:       status of the data from design to use as case
    control:    status of the data from design to use as control
    outfile:    Name of ouptut file (.csv)
    '''
    # Import R libraries
    base = importr('base')
    stats = importr('stats')
    limma = importr('limma')

    # Convert data and design pandas dataframes to R dataframes
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)
        r_design = ro.conversion.py2rpy(design)
        # Use the genes index column from data as a R String Vector
        genes = ro.StrVector([str(index) for index in data.index.tolist()])

    # Create a model matrix using design's Target column using the R formula "~0 + f"
    # to get all the unique factors as columns
    f = base.factor(r_design.rx2('Target'), levels=base.unique(r_design.rx2('Target')))
    form = Formula('~0 + f')
    form.environment['f'] = f
    r_design = stats.model_matrix(form)
    r_design.colnames = base.levels(f)

    # Fit the data to the design using lmFit from limma
    fit = limma.lmFit(r_data, r_design)
    # Make a contrasts matrix with the 1st and the last unique values
    contrast_matrix = limma.makeContrasts(
        f"{case}-{control}", levels=r_design)

    # Fit the contrasts matrix to the lmFit data & calculate the bayesian fit
    fit2 = limma.contrasts_fit(fit, contrast_matrix)
    fit2 = limma.eBayes(fit2)

    # topTreat the bayesian fit using the contrasts and add the genelist
    r_output = limma.topTreat(fit2, coef=1, genelist=genes, number=np.Inf)
    with localconverter(ro.default_converter + pandas2ri.converter):
        output = ro.conversion.rpy2py(r_output)
    output.to_csv(outfile, index=False)


if __name__ == '__main__':
    data_ = pd.read_excel('expression_file.xlsx')  # replace your own data file
    data_ = data_.set_index('ID')  # replace 'ID' with your own annotation if necessary
    design_ = pd.read_excel('limma_design_file.xlsx')  # replace with your own design file
    run_limma(data_, design_, "limma_output.xlsx")
