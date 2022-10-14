#!/usr/bin/env python

import os,sys,math,glob,ROOT,h5py
import numpy as np
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree, gDirectory, TChain, TH2D


def main(argv):
    gROOT.SetBatch(True)

    nvar = 6 #number of stored variables for each hit (includes event, volume, layer and module number)

    indir = "/global/cfs/cdirs/atlas/jmw464/mlkf_data/data/ttbar/ttbar_mu300/"
    outdir = "/global/homes/j/jmw464/ATLAS/KalmanML/data/"

    hits_file = TFile(indir+"hits.root")
    hits_tree = hits_file.Get("hits")
    measurements_file = TFile(indir+"measurements.root")

    outfile = h5py.File(outdir+"hits.hdf5","w")
  
    data = np.zeros((0,nvar))
    keylist = [key.GetName() for key in measurements_file.GetListOfKeys()]

    #read each folder in measurements file (corresponding to volume, layer and modules number)
    for key in keylist:
        vol = key.split("_")[0]
        layer_tree = measurements_file.Get(key)

        layer_data = np.zeros((layer_tree.GetEntries(),nvar))
        for ientry, meas_entry in enumerate(layer_tree):
            layer_data[ientry,0] = meas_entry.event_nr
            layer_data[ientry,1] = meas_entry.volume_id
            layer_data[ientry,2] = meas_entry.layer_id
            layer_data[ientry,3] = meas_entry.surface_id
            layer_data[ientry,4] = meas_entry.true_x

        data = np.append(data,layer_data,axis=0)

    #sort data for each volume based on event, volume, layer and module number (to match truth particles)
    data = data[data[:,3].argsort(kind='mergesort')] #sort by module
    data = data[data[:,2].argsort(kind='mergesort')] #sort by layer
    data = data[data[:,1].argsort(kind='mergesort')] #sort by volume
    data = data[data[:,0].argsort(kind='mergesort')] #sort by event

    #add on truth particle information from hits file - these are sorted by event, volume, layer and module
    for ientry, hit_entry in enumerate(hits_tree):
        data[ientry,5] = hit_entry.particle_id

    #write hdf5 file
    for i in range(data.shape[0]):
        event_id = int(data[i,0])
        volume_id = int(data[i,1])
        layer_id = int(data[i,2])

        groupname = str(event_id)+"/"+str(volume_id)+"/"+str(layer_id) #group structure is event/volume/layer
        if groupname not in outfile.keys():
            group = outfile.create_group(groupname)
            data_slice = data[np.logical_and(data[:,0]==event_id,np.logical_and(data[:,1]==volume_id,data[:,2]==layer_id))] #slice data corresponding to event, volume and layer
            group.create_dataset("hits",data=data_slice[:,3:]) #remove event, volume and layer information from array

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
