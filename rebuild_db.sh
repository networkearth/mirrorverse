#!/bin/bash

export DATABASE_URL="sqlite:////workspaces/mirrorverse/mirrorverse.db"

# Drop the database
rm mirrorverse.db

# build the fact tables
dvc repro -Rf pipelines/upload/tag_tracks/
dvc repro -Rf pipelines/upload/home_regions/
dvc repro -Rf pipelines/upload/surface_temperature/
dvc repro -Rf pipelines/upload/elevation/

# build the tags table
echo "missing_dimensions: ../tag_tracks/missing_dimensions.json" > pipelines/upload/tags/params.yaml
echo "data: data/HMM.Inventory_CSV_Marcel_2.12.2024.csv" >> pipelines/upload/tags/params.yaml
dvc repro -R pipelines/upload/tags/

echo "missing_dimensions: ../home_regions/missing_dimensions.json" > pipelines/upload/tags/params.yaml
echo "data: data/HMM.Inventory_CSV_Marcel_2.12.2024.csv" >> pipelines/upload/tags/params.yaml
dvc repro -R pipelines/upload/tags/

# build the spatial table
echo "missing_dimensions: ../tag_tracks/missing_dimensions.json" > pipelines/upload/spatial/params.yaml
dvc repro -R pipelines/upload/spatial/

echo "missing_dimensions: ../tags/missing_dimensions.json" > pipelines/upload/spatial/params.yaml
dvc repro -R pipelines/upload/spatial/

echo "missing_dimensions: ../surface_temperature/missing_dimensions.json" > pipelines/upload/spatial/params.yaml
dvc repro -R pipelines/upload/spatial/

echo "missing_dimensions: ../elevation/missing_dimensions.json" > pipelines/upload/spatial/params.yaml
dvc repro -R pipelines/upload/spatial/

# build the dates table
echo "missing_dimensions: ../tag_tracks/missing_dimensions.json" > pipelines/upload/dates/params.yaml
dvc repro -R pipelines/upload/dates/

echo "missing_dimensions: ../tags/missing_dimensions.json" > pipelines/upload/dates/params.yaml
dvc repro -R pipelines/upload/dates/

echo "missing_dimensions: ../surface_temperature/missing_dimensions.json" > pipelines/upload/dates/params.yaml
dvc repro -R pipelines/upload/dates/
