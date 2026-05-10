# FUNSD Dataset

## Source
- Project page: <https://guillaumejaume.github.io/FUNSD/>
- Direct download: <https://guillaumejaume.github.io/FUNSD/dataset.zip>
- Paper: Jaume G, Kemal Ekenel H, Thiran J. FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents. ICDAR Workshops 2019. DOI:10.1109/ICDARW.2019.10029

## Already downloaded
The dataset (~16 MB) has been fetched and extracted into this folder during implementation:

```
data/
└── dataset/
    ├── training_data/
    │   ├── annotations/   (149 .json files, one per form)
    │   └── images/        (149 .png files)
    └── testing_data/
        ├── annotations/   (50 .json files)
        └── images/        (50 .png files)
```

## Re-download from scratch
If `data/dataset/` is missing, run:

```bash
cd data/
curl -sLo funsd.zip 'https://guillaumejaume.github.io/FUNSD/dataset.zip'
unzip -q funsd.zip
rm funsd.zip
rm -rf __MACOSX
```

## Annotation schema

Each `annotations/<form_id>.json` has shape:

```json
{
  "form": [
    {
      "box": [x0, y0, x1, y1],
      "text": "<entity-level text>",
      "label": "question | answer | header | other",
      "words": [{"box": [x0, y0, x1, y1], "text": "<word>"}],
      "linking": [[src_id, dst_id], ...],
      "id": <int>
    },
    ...
  ]
}
```

- `box`: entity-level bounding box, pixel coordinates relative to the form image (typically 754x1000).
- `words`: word-level boxes and text inside the entity. Use these as the unit of token classification.
- `label`: one of four classes - `question`, `answer`, `header`, `other`.
- `linking`: list of `[source_id, destination_id]` pairs that connect a question to its answer (or a header to its block).
- `id`: integer id, unique within the form, used as the node id in the linking graph.

## Splits
- Train: 149 forms (`training_data/`)
- Test: 50 forms (`testing_data/`)
- The FUNSD paper does not define an official validation split. The baseline and advanced scripts split a stratified 10% off `training_data` for validation (set seed in script).

## Licence
FUNSD is a subset of the RVL-CDIP dataset (subset of IIT-CDIP Test Collection, Lewis et al. 2006). Released under a research-only licence; check the project page for the exact terms before redistribution.
