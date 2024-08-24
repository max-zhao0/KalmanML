#!/bin/bash

DATA_DIR=/global/cfs/cdirs/atlas/jmw464/mlkf_data/new
DATASET=muon50_100

echo "Working on ${DATASET}..."

echo "Processing hits"
python scripts/process/import_hits.py --data_dir ${DATA_DIR}/${DATASET}/ --out_dir ${DATA_DIR}/${DATASET}/processed/

echo "Processing truth particles"
python scripts/process/import_particles.py --data_dir ${DATA_DIR}/${DATASET}/ --out_dir ${DATA_DIR}/${DATASET}/processed/

echo "Processing tracks"
python scripts/process/import_tracks.py --data_dir ${DATA_DIR}/${DATASET}/ --out_dir ${DATA_DIR}/${DATASET}/processed/