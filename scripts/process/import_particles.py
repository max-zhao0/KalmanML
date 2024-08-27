#!/usr/bin/env python

import os, sys, h5py, argparse
from tqdm import tqdm
import numpy as np
from ROOT import gROOT, TFile


def main(argv):
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help = 'Path containing data generated with ACTS (with particles_initial.root file)')
    parser.add_argument('--out_dir', type=str, help = 'Path to store processed data')
    parser.add_argument('-o', '--overwrite', action='store_true', help = 'Overwrite existing output file')
    args = parser.parse_args()
    
    particles_file = TFile(args.in_dir+"particles_simulation.root")

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.overwrite or not os.path.exists(args.out_dir+"particles.hdf5"):
        outfile = h5py.File(args.out_dir+"particles.hdf5","w")
    else:
        raise FileExistsError("Output file particles.hdf5 already exists. Use -o to overwrite")
    
    particles_tree = particles_file.Get("particles")

    particles_dtypes = np.dtype([('particle_id', '<i4'), ('eta', '<f4'), ('phi', '<f4'), ('pt', '<f4')])

    for nevent, event_particles in enumerate(tqdm(particles_tree)):
        nparticles = len(getattr(event_particles, "eta"))
        particles_data = np.empty(nparticles, dtype=particles_dtypes)
        for name in particles_dtypes.names:
            particles_data[name] = getattr(event_particles, name)
        
        group = outfile.create_group(str(nevent))
        group.create_dataset("particles", data=particles_data)

    outfile.close()


if __name__ == "__main__":
    main(sys.argv)