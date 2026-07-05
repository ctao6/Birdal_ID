"""
Loading and querying the two reference CSVs:
  - species_colors.csv: one row per species/illustration, columns include
    identifying info (Com_name, Sci_name, etc.) plus color1..color24 percentages.
  - color_clusters.csv: 24 rows, one per cluster, with R, G, B columns defining
    the fixed cluster center each color1..color24 column corresponds to.

Author: Christine Tao
Date: 7/4/2026
"""

import pandas as pd
import numpy as np

def load_cluster_centers(path: str) -> np.ndarray:
    """
    Load the 24 cluster-center RGB values.
 
    Returns:
        np.ndarray of shape (24, 3) -- one RGB triplet per cluster, in the same
        order as color1..color24 in the species table. Careful: double check the
        CSV's row order actually matches color1->color24 order, or sort/reindex
        explicitly by the 'Color classification' column so you don't silently
        mismatch cluster indices with the species table's columns.
    """

    df = pd.read_csv(path)
    df['cluster_index'] = df['Color classification'].apply(lambda s: int(s[5:]))
    df = df.sort_values(by = 'cluster_index')
    ret = df.loc[:, 'R':'B']

    return ret.to_numpy()

def load_species_vectors(path: str) -> "tuple[np.ndarray, pd.DataFrame]":
    """
    Load the species reference table.
 
    Returns:
        vectors: np.ndarray of shape (num_species_rows, 24) -- the color1..color24
                 percentage columns only, as floats.
        metadata: pd.DataFrame with the remaining identifying columns (Com_name,
                  Sci_name, eBird species code, etc.), same row order as `vectors`,
                  so you can map a matched row index back to a readable species name.
    """
    df = pd.read_csv(path)
    vector = (df.loc[:,'color1':'color24']).to_numpy()
    metadata = df.drop(columns=df.loc[:, 'color1':'color24'].columns)


    return (vector, metadata)    

    

                                                           
                                                        
