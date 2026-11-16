# Data

Neither dataset is redistributed in this repository. Download both from their original sources and accept their terms.

## DENTEX (source dataset)

- Source: DENTEX Challenge 2023, Zenodo record [7812323](https://zenodo.org/records/7812323), file `training_data.zip`.
- Licence: CC BY 4.0.
- Used part: the quadrant-enumeration-diagnosis subset (705 publicly labeled panoramic radiographs; the challenge validation and test labels are not public). The quadrant-enumeration subset is read only by the Stage G validity check.

Expected layout after unzipping:

```
DENTEX/training_data/
├── quadrant-enumeration-disease/     <- DENTEX_DIR
│   ├── xrays/                        panoramic radiographs
│   └── *.json                        COCO-style annotations with category_id_1/2/3
└── quadrant_enumeration/
    ├── xrays/
    └── train_quadrant_enumeration.json
```

The notebook finds the diagnosis JSON by looking for annotations that carry `category_id_3`. Diagnosis annotations that share a radiograph, quadrant and tooth number are merged into one multi-label tooth. In the release we used, 678 of the 705 radiographs carry at least one diagnosis annotation, giving 3,456 teeth.

## ADCD (target dataset)

- Source: Zenodo, doi [10.5281/zenodo.15487430](https://doi.org/10.5281/zenodo.15487430).
- Licence: database usage agreement restricting use to non-commercial research and education.
- Used part: the YOLO-format panoramic dataset with 14 classes. Classes 4, 6 and 7 are mapped to caries, impacted tooth and periapical lesion. The notebook checks these indices against the class names when a `data.yaml` is present.

Expected layout:

```
panoramic_radiography_yolo_dataset_14_classes/   <- ADCD_DIR
├── train/  {images/, labels/}
├── valid/  {images/, labels/}
├── test/   {images/, labels/}
└── test_alte_cabinete/Ext-validation/  {images/, labels/}
```

The train, valid and test folders are pooled and re-split by radiograph for each seed. The external folder is held out from all training, labeled slices and gate fitting. In the release we used it contains 136 radiographs, while the dataset description reports 180.

The release contains 136 files in the internal folders that are byte-identical copies of other internal files. The notebook groups identical files by content hash so that a radiograph and its copy always fall in the same split, and reports the count in `adcd_parse_stats.json`. No external radiograph appears in the internal folders.

## Crops

DENTEX boxes outline teeth. ADCD boxes outline lesions and are smaller, so each ADCD box is first expanded about its center to at least the median DENTEX tooth box (169 × 288 px). Both datasets then receive a 15% margin on each side before embedding.
