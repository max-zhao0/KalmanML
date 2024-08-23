#!/bin/bash

ACTS_DIR=/global/cfs/cdirs/atlas/jmw464/Software/acts
OUT_DIR=/global/cfs/cdirs/atlas/jmw464/mlkf_data/new
NEVENTS=100

export LD_LIBRARY_PATH=${ACTS_DIR}/build/thirdparty/OpenDataDetector/factory:${LD_LIBRARY_PATH}
source ${ACTS_DIR}/CI/setup_cvmfs_lcg.sh
source ${ACTS_DIR}/build/python/setup.sh


if [[ $1 == "ttbar" ]]; then
    echo "Generating ${NEVENTS} ttbar events"
    python acts/generate_pp.py --acts_dir ${ACTS_DIR} --out_dir ${OUT_DIR} --n_events ${NEVENTS}
elif [[ $1 == "muon" ]]; then
    echo "Generating ${NEVENTS} muon events"
    python acts/generate_muons.py --acts_dir ${ACTS_DIR} --out_dir ${OUT_DIR} --n_events ${NEVENTS}
else
    echo "Please provide valid mode: ttbar or muon"
fi

