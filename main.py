"""
Entry point: run the full pipeline on a single test image.
 
Author: Christine Tao
Date: 7/4/2026
"""
import pandas as pd

from bird_id.reference_data import load_cluster_centers, load_species_vectors
from bird_id.image_processing import load_opaque_pixels
from bird_id.color_vector import image_color_vector
from bird_id.matching import rank_species

load = (pd.read_csv('data/LikelyBirds.csv')).head(50)
plausible_species = load['Common Name'].tolist()

pixels = load_opaque_pixels('test.png', 1)
cluster_centers = load_cluster_centers('data/color_clusters.csv')
species_vectors, species_metadata = load_species_vectors('data/species_colors.csv')
image_vector = image_color_vector(pixels, cluster_centers, 0.1)
rank = rank_species(image_vector, species_vectors, species_metadata, 18750, "cosine")

ret = rank[rank['Com_name'].isin(plausible_species)]

print(ret.head(10))
