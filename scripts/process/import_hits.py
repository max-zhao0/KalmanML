#!/usr/bin/env python

import os, sys, h5py, uproot, argparse
from tqdm import tqdm
import numpy as np
import awkward as ak
from ROOT import gROOT


def main(argv):
    gROOT.SetBatch(True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help = 'Path containing data generated with ACTS (with hits.root and measurements.root files)')
    parser.add_argument('--out_dir', type=str, help = 'Path to store processed data')
    parser.add_argument('-o', '--overwrite', action='store_true', help = 'Overwrite existing output file')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.overwrite or not os.path.exists(args.out_dir+"hits.hdf5"):
        outfile = h5py.File(args.out_dir+"hits.hdf5","w")
    else:
        raise FileExistsError("Output file hits.hdf5 already exists. Use -o to overwrite")

    hits_file = uproot.open(args.in_dir+"hits.root")
    meas_file = uproot.open(args.in_dir+"measurements.root")
    hits_tree = hits_file["hits"]
    meas_tree = meas_file["measurements"]

    nentries = meas_tree.num_entries
    assert nentries == hits_tree.num_entries, "Number of entries in hits and measurements trees do not match"

    meas_dtypes = np.dtype([('event_nr', '<i4'), ('volume_id', '<i2'), ('layer_id', '<i2'), ('surface_id', '<i2'), ('rec_loc0', '<f4'), ('rec_loc1', '<f4'), ('channel_value', '<f4'), ('clus_size', '<f4'), ('true_x', '<f4'), ('true_y', '<f4'), ('true_z', '<f4'), ('particle_id', '<i4')])
    hits_dtypes = np.dtype([('event_id', '<i4'), ('tx', '<f4'), ('ty', '<f4'), ('tz', '<f4'), ('volume_id', '<i2'), ('layer_id', '<i2'), ('sensitive_id', '<i2'), ('particle_id', '<i4')])

    meas_data = np.empty(nentries, dtype=meas_dtypes)
    hits_data = np.empty(nentries, dtype=hits_dtypes)
    
    #read in measurements from file
    for name in meas_dtypes.names:
        if name == "channel_value":
            meas_data[name] = ak.sum(meas_tree[name].array(library="ak"), axis=1)
        elif name == "particle_id": #particle_id will be set afterwards
            continue
        else:
            meas_data[name] = meas_tree[name].array(library="np")

    #read in hits from file
    for name in hits_dtypes.names:
        hits_data[name] = hits_tree[name].array(library="np")

    print("Processing hit information")

    curr_event = -1
    for i in tqdm(range(nentries)):
        if curr_event != meas_data["event_nr"][i]:
            curr_event = meas_data["event_nr"][i]
            event_truth = hits_data[meas_data["event_nr"][i] == hits_data["event_id"]]
        match_index = np.argwhere(np.logical_and.reduce([
            meas_data["true_x"][i] == event_truth["tx"],
            meas_data["true_y"][i] == event_truth["ty"]
        ]))[0]
        meas_data["particle_id"][i] = event_truth["particle_id"][match_index]

    print("Restructuring and writing hits to file")

    #write hdf5 file
    for i in tqdm(range(nentries)):
        event_id = int(meas_data["event_nr"][i])
        volume_id = int(meas_data["volume_id"][i])

        groupname = str(event_id)+"/"+str(volume_id) #group structure is event/volume/layer/module

        if groupname not in outfile.keys():
            group = outfile.create_group(groupname)
            data_slice = meas_data[np.logical_and(meas_data["event_nr"]==event_id, meas_data["volume_id"]==volume_id)] #slice data corresponding to event, volume, layer and module
            group.create_dataset("hits",data=data_slice) #remove event, volume and layer information from array

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
