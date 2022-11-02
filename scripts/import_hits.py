#!/usr/bin/env python

import os,sys,math,glob,ROOT,h5py
import numpy as np
import awkward as ak
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree, gDirectory, TChain, TH2D
import uproot


def main(argv):
    gROOT.SetBatch(True)

    nvar = 12 #number of stored variables for each hit (includes event, volume, layer and module number)

    indir = "/global/cfs/cdirs/atlas/jmw464/mlkf_data/data/ttbar/ttbar_mu300/"
    outdir = "/global/homes/j/jmw464/ATLAS/KalmanML/data/"

    hits_file = TFile(indir+"hits.root")
    hits_tree = hits_file.Get("hits")
    measurements_file = uproot.open(indir+"measurements.root")

    outfile = h5py.File(outdir+"hits.hdf5","w")

    data = np.zeros((0,nvar))
    keylist = measurements_file.keys()

    #read each folder in measurements file (corresponding to volume, layer and modules number)
    for key in keylist:
        vol = key.split("_")[0]
        layer_tree = measurements_file[key]
        
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

    #sort data for each volume based on event, volume, layer and module number (to match truth particles)
    data = data[data[:,3].argsort(kind='mergesort')] #sort by module
    data = data[data[:,2].argsort(kind='mergesort')] #sort by layer
    data = data[data[:,1].argsort(kind='mergesort')] #sort by volume
    data = data[data[:,0].argsort(kind='mergesort')] #sort by event

    #add on truth particle information from hits file - these are sorted by event, volume, layer and module
    for ientry, hit_entry in enumerate(hits_tree):
        data[ientry,-1] = hit_entry.particle_id

    #write hdf5 file
    for i in range(data.shape[0]):
        event_id = int(data[i,0])
        volume_id = int(data[i,1])
        layer_id = int(data[i,2])
        module_id = int(data[i,3])

        groupname = str(event_id)+"/"+str(volume_id)+"/"+str(layer_id)+"/"+str(module_id) #group structure is event/volume/layer/module
        if groupname not in outfile.keys():
            group = outfile.create_group(groupname)
            data_slice = data[np.logical_and.reduce((data[:,0]==event_id,data[:,1]==volume_id,data[:,2]==layer_id,data[:,3]==module_id))] #slice data corresponding to event, volume, layer and module
            group.create_dataset("hits",data=data_slice[:,4:]) #remove event, volume and layer information from array

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
