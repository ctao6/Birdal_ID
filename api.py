"""
FAST API

Author: Christine Tao
Date: 7/5/2026
"""

#module imports
from fastapi import FastAPI, UploadFile, Form
from datetime import date
import io
import json
import pandas

#function imports for id pipeline
from bird_id.reference_data import load_cluster_centers, load_species_vectors
from bird_id.image_processing import load_opaque_pixels
from bird_id.color_vector import image_color_vector
from bird_id.matching import rank_species

app = FastAPI()

cluster_centers = load_cluster_centers('data/color_clusters.csv')
species_vectors, species_metadata = load_species_vectors('data/species_colors.csv')
species_metadata = species_metadata.drop(columns = ['Morph','Illustration_id','Credit','Age'])

@app.post("/identify")
async def identify_bird(img: UploadFile, longitude: float = Form(...), latitude: float = Form(...), date: date = Form(...), plausible_species_json: str = Form(...)):
    raw_bytes = await img.read()
    image_stream = io.BytesIO(raw_bytes)
    print(repr(plausible_species_json))
    plausible_species = json.loads(plausible_species_json) 

    pixels = load_opaque_pixels(image_stream, 1)

    image_vector = image_color_vector(pixels, cluster_centers, 0.1)
    rank = rank_species(image_vector, species_vectors, species_metadata, 18750, "cosine")

    filtered = (rank[rank['Com_name'].isin(plausible_species)]).head(10)

    return filtered.to_dict(orient = 'records')

    
