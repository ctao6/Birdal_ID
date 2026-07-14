## Birdal ID

A color-based bird species identification service. Given a user-drawn sketch of a bird plus a location and date, it returns a ranked list of likely species.

This service is the matching "brain" behind the Birdal app's bird ID feature.

**How it works**

Traditional image classifiers need hundreds of labeled examples per class to train which was feasible for a young app with a handful of user-submitted drawings per species. Instead of training a classifier, this service compares color distributions within the submitted drawing.

1. **Reference data**: [a public dataset](https://datadryad.org/dataset/doi:10.5061/dryad.70rxwdc6s) quantifies plumage color across 18,757 illustrations from the *Handbook of the Birds of the World*, clustering each illustration's pixels into 24 fixed color buckets and recording what percentage of the image each bucket occupies.
2. **User drawings** are processed the same way: every non-transparent pixel is softly assigned to the same 24 color buckets (an exponential-decay weighting based on distance to each bucket's reference color), then averaged into one 24-dimensional vector for the whole image.
3. Because both sides — reference illustrations and user drawings — end up as 24-length vectors that sum to 1, they're directly comparable with cosine or Euclidean distance.
4. The result is ranked by similarity, optionally restricted to a caller-supplied list of plausible species (used upstream to filter by location/season via eBird).

This approach sidesteps needing a large labeled training set, at the cost of being purely color-based — it has no notion of shape, size, or where a color sits on the bird's body.

## Project structure

```
bird_id_project/
├── data/
│   ├── color_clusters.csv    # 24 reference RGB cluster centers
│   └── species_colors.csv    # per-species/illustration color vectors + metadata
├── bird_id/
│   ├── reference_data.py     # loading the two CSVs above
│   ├── image_processing.py   # PNG -> opaque RGB pixel array
│   ├── color_vector.py       # pixel(s) -> 24-dim color vector
│   └── matching.py           # compare a vector against the reference table, rank
├── api.py                    # FastAPI service wrapping the pipeline above
├── main.py                   # local CLI entry point for one-off testing
├── requirements.txt
└── .python-version
```

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 -m uvicorn api:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs, or test directly:

```bash
curl -X POST http://127.0.0.1:8000/identify \
  -F "img=@test.png" \
  -F "latitude=42.6" \
  -F "longitude=-73.5" \
  -F "date=2026-01-15" \
  -F 'plausible_species_json=["American Robin", "Rock Pigeon"]'
```

## API

### `POST /identify`

| Field | Type | Description |
|---|---|---|
| `img` | file | PNG drawing, transparent background |
| `latitude` | float | |
| `longitude` | float | |
| `date` | date (`YYYY-MM-DD`) | |
| `plausible_species_json` | string | JSON array of common names to restrict ranking to (typically supplied by the caller from a recent eBird lookup) |

Returns the top 10 matching illustrations as JSON, each including `Com_name`, `Sci_name`, and a `score` (lower = closer match).

## Deployment

Deployed on Render as a standard Python web service. Build command `pip install -r requirements.txt`, start command `uvicorn api:app --host 0.0.0.0 --port $PORT`. A `.python-version` file pins the Python version to avoid pulling in packages that require compiling from source.

## Known limitations / next steps

- **Color-only matching** struggles with species that share a similar coarse color palette but differ mainly in shape or the *location* of color on the body (e.g. Rock Pigeon vs. other gray-toned birds) — a spatial or learned model is the natural next step once enough labeled user drawings exist.
- Location/date filtering currently relies on eBird's "recent observations nearby" endpoint (actual sightings in the last N days) rather than true historical frequency data, as a simpler alternative to eBird's region-code-based bar chart data.
