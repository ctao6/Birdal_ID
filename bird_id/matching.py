"""
Comparing one image's color vector against every species in the reference
table, and returning the top N most similar species.
"""
 
import numpy as np
import pandas as pd
 
 
def rank_species(
    image_vector: np.ndarray,
    species_vectors: np.ndarray,
    species_metadata: pd.DataFrame,
    top_n: int = 5,
    metric: str = "cosine",
) -> pd.DataFrame:
    """
    Rank all species by similarity to the given image color vector.
 
    Args:
        image_vector: shape (24,) -- from color_vector.image_color_vector().
        species_vectors: shape (num_species_rows, 24) -- from
                          reference_data.load_species_vectors().
        species_metadata: DataFrame aligned row-for-row with species_vectors,
                           containing identifying info (Com_name, Sci_name, etc.)
        top_n: how many top matches to return.
        metric: "cosine" or "euclidean" -- which distance metric to use.
                (Recall: we discussed these should behave similarly here, since
                all vectors sum to 1, but useful to support both for comparison.)
 
    Returns:
        A copy of species_metadata's top_n rows (sorted best-match-first), with
        an added "score" column indicating similarity/distance to the image.
    """
    from scipy.spatial.distance import cdist
    distances = cdist(image_vector.reshape(1, 24), species_vectors, metric=metric)

    top_indices = np.argsort(distances[0])[:top_n]
    ret = species_metadata.iloc[top_indices].copy()
    ret['score'] = distances[0][top_indices]

    return ret
