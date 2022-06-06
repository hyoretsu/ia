import numpy as np
import seaborn as sns
from pandas import DataFrame


def drop_correlation(dataset: DataFrame, threshold: int):
    # Create correlation matrix
    corr_matrix = dataset.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    return dataset.drop(to_drop, axis=1)


lukewarm_cmap = sns.diverging_palette(240, 10, as_cmap=True)
