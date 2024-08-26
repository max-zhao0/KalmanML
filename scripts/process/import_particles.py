#!/usr/bin/env python

import sys, h5py, argparse
from ROOT import gROOT, TFile
import numpy as np


def main(argv):
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help = 'Path containing data generated with ACTS (with particles_initial.root file)')
    parser.add_argument('--out_dir', type=str, help = 'Path to store processed data')
    args = parser.parse_args()
    
    infile = TFile(args.in_dir + "/particles_initial.root")
    part_tree = infile.Get("particles")

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    outfile = h5py.File(args.out_dir + "particles.hdf5", "w")

    for event_id, event_particles in enumerate(part_tree):
        data = np.empty((len(event_particles.particle_id), 4))
        data[:,0] = event_particles.particle_id
        data[:,1] = event_particles.eta
        data[:,2] = event_particles.phi
        data[:,3] = event_particles.pt
        
        group = outfile.create_group(str(event_id))
        group.create_dataset("particles", data=data)

    outfile.close()

    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))