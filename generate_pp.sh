#!/bin/bash

nEvents=50
nPileups=(200)

ckf_chi2max=30
ckf_nmax=15

DATA_DIR=/global/cfs/cdirs/atlas/jmw464/mlkf_data/shell
ACTS_INSTALL=/global/homes/j/jmw464/ATLAS/Software/acts19.2/build
ACTS_HOME=/global/homes/j/jmw464/ATLAS/Software/acts19.2

for n in "${nPileups[@]}"
do
    sub_dir=ttbar${n}_${nEvents}
    
    command_particle_gun="
    ${ACTS_INSTALL}/bin/ActsExamplePythia8 \
    --events=${nEvents} \
    --output-dir=${DATA_DIR}/${sub_dir} \
    --output-root \
    --output-csv \
    --rnd-seed=42 \
    --gen-cms-energy-gev=14000 \
    --gen-hard-process=Top:qqbar2ttbar=on \
    --gen-npileup=${n}"

    ${command_particle_gun}
    
    command_fatras="
    ${ACTS_INSTALL}/bin/ActsExampleFatrasGeneric \
    --events=${nEvents} \
    --input-dir=${DATA_DIR}/${sub_dir} \
    --output-dir=${DATA_DIR}/${sub_dir}  \
    --output-root \
    --output-csv \
    --select-eta=-3:3 \
    --select-pt=0.5: \
    --remove-neutral \
    --bf-constant-tesla=0:0:2 \
    "
    
    ${command_fatras}
    
    command_digitization="
    ${ACTS_INSTALL}/bin/ActsExampleDigitizationGeneric \
    --events=${nEvents} \
    --input-dir=${DATA_DIR}/${sub_dir} \
    --output-dir=${DATA_DIR}/${sub_dir} \
    --output-root \
    --output-csv \
    --digi-config-file ${ACTS_HOME}/Examples/Algorithms/Digitization/share/default-geometric-config-generic.json \
    --bf-constant-tesla=0:0:2 \
    "
    
    ${command_digitization}

    command_seeding="
    ${ACTS_INSTALL}/bin/ActsExampleSeedingGeneric \
    --events=${nEvents} \
    --input-dir=${DATA_DIR}/${sub_dir} \
    --output-dir=${DATA_DIR}/${sub_dir} \
    --bf-constant-tesla=0:0:2 \
    --digi-merge \
    --digi-config-file ${ACTS_HOME}/Examples/Algorithms/Digitization/share/default-smearing-config-generic.json  \
    --geo-selection-config-file ${ACTS_HOME}/Examples/Algorithms/TrackFinding/share/geoSelection-genericDetector.json 
    "

    ${command_seeding}

    command_kalman="
    ${ACTS_INSTALL}/bin/ActsExampleCKFTracksGeneric \
    --input-dir=${DATA_DIR}/${sub_dir} \
    --output-dir=${DATA_DIR}/${sub_dir} \
    --bf-constant-tesla=0:0:2 \
    --ckf-selection-chi2max=${ckf_chi2max} \
    --ckf-selection-nmax=${ckf_nmax} \
    --digi-config-file ${ACTS_HOME}/Examples/Algorithms/Digitization/share/default-smearing-config-generic.json  \
    --geo-selection-config-file ${ACTS_HOME}/Examples/Algorithms/TrackFinding/share/geoSelection-genericDetector.json \
    "

    ${command_kalman}

    command_truth="
    ${ACTS_INSTALL}/bin/ActsExampleTruthTracksGeneric \
    --input-dir=${DATA_DIR}/${sub_dir} \
    --output-dir=${DATA_DIR}/${sub_dir} \
    --bf-constant-tesla=0:0:2 \
    --digi-config-file ${ACTS_HOME}/Examples/Algorithms/Digitization/share/default-smearing-config-generic.json  \
    "

    ${command_truth}

done
