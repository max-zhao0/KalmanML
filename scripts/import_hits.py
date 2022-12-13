#!/usr/bin/env python

import os,sys,math,glob,ROOT,h5py
import numpy as np
import awkward as ak
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree, gDirectory, TChain, TH2D
import uproot
import time


def main(argv):
    gROOT.SetBatch(True)

    nvar = 12 #number of stored variables for each hit (includes event, volume, layer and module number)

    indir = "/global/cfs/cdirs/atlas/jmw464/mlkf_data/shell/ttbar200_50/"
    outdir = indir+"processed/"

    if not os.path.exists(outdir): os.mkdir(outdir)

    hits_file = uproot.open(indir+"hits.root")
    hits_tree = hits_file["hits"]
    measurements_file = uproot.open(indir+"measurements.root")

    outfile = h5py.File(outdir+"hits.hdf5","w")

    data = np.zeros((0,nvar))
    keylist = measurements_file.keys()
    
    for i, key in enumerate(keylist):
        keylist[i] = key.split(";")[0]
    keylist = list(set(keylist))
    keylist.sort()

    start_time = time.time()
    print("Beginning data processing")

    #read each folder in measurements file (corresponding to volume, layer and modules number)
    measurement_entries = 0
    for key in keylist:
        cur_time = time.time()
        vol = key.split("_")[0]
        layer_tree = measurements_file[key]

        measurement_entries += layer_tree.num_entries
        layer_data = np.zeros((layer_tree.num_entries,nvar))
        
        layer_data[:,0] = layer_tree["event_nr"].array(library="np")
        layer_data[:,1] = layer_tree["volume_id"].array(library="np")
        layer_data[:,2] = layer_tree["layer_id"].array(library="np")
        layer_data[:,3] = layer_tree["surface_id"].array(library="np")
        layer_data[:,4] = layer_tree["rec_loc0"].array(library="np")
        layer_data[:,5] = layer_tree["rec_loc1"].array(library="np")
        layer_data[:,6] = ak.sum(layer_tree["channel_value"].array(library="ak"),axis=1)
        layer_data[:,7] = layer_tree["clus_size"].array(library="np")
        layer_data[:,8] = layer_tree["true_x"].array(library="np")
        layer_data[:,9] = layer_tree["true_y"].array(library="np")
        layer_data[:,10] = layer_tree["true_z"].array(library="np")

        data = np.append(data,layer_data,axis=0)

        print("Processed key {} with {} hits in {} seconds".format(key, layer_tree.num_entries, time.time()-cur_time))

    #sort data for each volume based on event, volume, layer and module number (to match truth particles)
    data = data[data[:,3].argsort(kind='mergesort')] #sort by module
    data = data[data[:,2].argsort(kind='mergesort')] #sort by layer
    data = data[data[:,1].argsort(kind='mergesort')] #sort by volume
    data = data[data[:,0].argsort(kind='mergesort')] #sort by event

    print("Adding truth particle info. Time elapsed: {} seconds".format(time.time()-start_time))
    
    #read truth particle information from hits file
    truth_data = np.zeros((hits_tree.num_entries,8))
    truth_data[:,0] = hits_tree["event_id"].array(library="np")
    truth_data[:,1] = hits_tree["tx"].array(library="np")
    truth_data[:,2] = hits_tree["ty"].array(library="np")
    truth_data[:,3] = hits_tree["tz"].array(library="np")
    truth_data[:,4] = hits_tree["volume_id"].array(library="np")
    truth_data[:,5] = hits_tree["layer_id"].array(library="np")
    truth_data[:,6] = hits_tree["sensitive_id"].array(library="np")
    truth_data[:,7] = hits_tree["particle_id"].array(library="np")
    truth_data = truth_data[truth_data[:,0].argsort(kind="mergesort")] #sort by event
  
    match_array = np.logical_and(np.in1d(truth_data[:,1], data[:,8]), np.in1d(truth_data[:,2], data[:,9])) #match hits from hits file to hits from measurements file
    truth_data = truth_data[match_array]
    data[:,-1] = truth_data[:,-1]

    print("Writing file. Time elapsed: {} seconds".format(time.time()-start_time))

    #write hdf5 file
    cur_time = time.time()
    for i in range(data.shape[0]):
        event_id = int(data[i,0])
        volume_id = int(data[i,1])

        groupname = str(event_id)+"/"+str(volume_id) #group structure is event/volume/layer/module

        if groupname not in outfile.keys():
            group = outfile.create_group(groupname)
            data_slice = data[np.logical_and(data[:,0]==event_id,data[:,1]==volume_id)] #slice data corresponding to event, volume, layer and module
            group.create_dataset("hits",data=data_slice[:,2:]) #remove event, volume and layer information from array
            print("Writing {} in {} seconds".format(groupname, time.time()-cur_time))
            cur_time = time.time()

    print("Finished. Time elapsed: {} seconds".format(time.time()-start_time))

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
