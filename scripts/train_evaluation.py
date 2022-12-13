#!/usr/bin/env python

import os,sys,math,glob,ROOT,h5py
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from ROOT import gROOT, TFile, TH1D, TLorentzVector, TCanvas, TTree, gDirectory, TChain, TH2D

from evaluation_model import *


def main(argv):
    gROOT.SetBatch(True)

    use_gpu = True
    load_checkpoint = True
    freeze_linear = False
    freeze_transformer = False

    runnumber = 0
    nepochs = 10
    batch_size = 15
    train_prop = 0.8
    val_prop = 0.1
    max_len = 25
    class_threshold = 0.5

    indir = "/global/cfs/cdirs/atlas/jmw464/mlkf_data/data/processed/"
    outdir = "/global/homes/j/jmw464/ATLAS/KalmanML/data/"

    #---------------------------- IMPORT DATA ----------------------------

    tracks_file = h5py.File(indir+"tracks.hdf5","r")

    track_truth = tracks_file["tracks"]["truth"]
    track_meas = tracks_file["tracks"]["measurements"]

    ntracks = track_truth.shape[0]
    train_idx = round(ntracks*train_prop)
    val_idx = round(ntracks*(train_prop+val_prop))

    train_loader = th.utils.data.DataLoader([[track_meas[i], track_truth[i,:,:1]] for i in range(train_idx)], shuffle=False, batch_size=batch_size)
    val_loader = th.utils.data.DataLoader([[track_meas[i], track_truth[i,:,:1]] for i in range(train_idx, val_idx)], shuffle=False, batch_size=batch_size)
    test_loader = th.utils.data.DataLoader([[track_meas[i], track_truth[i,:,:1]] for i in range(val_idx, ntracks)], shuffle=False, batch_size=batch_size)

    #---------------------------- SET UP NETWORK ----------------------------
    
    if not os.path.exists(outdir+str(runnumber)):
        os.makedirs(outdir+str(runnumber))

    checkpointfile_name = outdir+str(runnumber)+"/transformer_"+str(runnumber)+"_model.pt"

    if th.cuda.is_available() and use_gpu:
        device = th.device('cuda')
        print("Found {} GPUs".format(th.cuda.device_count()))
    else:
        device = th.device('cpu')

    model = EvalNN(7,10,2,10,False,0.1,max_len).double().to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.BCELoss()

    #load existing checkpoint
    if load_checkpoint and os.path.exists(checkpointfile_name):
        checkpoint = th.load(checkpointfile_name, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        print("Loading previous model. Starting from epoch {}.".format(start_epoch), flush=True)
    else:
        start_epoch = 1


    #print model parameters
    print("Model built. Parameters:", flush=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
        if freeze_linear and "linear" in name:
            param.requires_grad = False
        if freeze_transformer and "transformer" in name:
            param.requires_grad = False
        print(name, param.size(), param.requires_grad, flush=True)
    print("", flush=True)

    if start_epoch == nepochs+1:
        print("Model already trained. Skipping training")

    #---------------------------- TRAIN NETWORK ----------------------------

    train_loss_array = np.zeros(nepochs)
    val_loss_array = np.zeros(nepochs)

    for epoch in range(start_epoch,nepochs+1):
        print("Epoch: {}".format(epoch), flush=True)

        for ibatch, data in enumerate(train_loader):
            batch, train_labels = data
            pred = model(batch)
            pred_lt = loss(pred, train_labels)
            train_loss_array[epoch-1] += pred_lt.item()

            optimizer.zero_grad()
            pred_lt.backward()
            optimizer.step()
        
        train_loss_array[epoch-1] = train_loss_array[epoch-1]/(train_idx+1)
        print("Training loss: {}".format(train_loss_array[epoch-1]), flush=True)

        #save checkpoint
        th.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpointfile_name)

        model.eval()
        for ibatch, data in enumerate(val_loader):
            batch, val_labels = data
            pred = model(batch)
            pred_lt = loss(pred, val_labels)
            val_loss_array[epoch-1] += pred_lt.item()
    
        val_loss_array[epoch-1] = val_loss_array[epoch-1]/(val_idx-train_idx+1)
        print("Validation loss: {}".format(val_loss_array[epoch-1]), flush=True)
        print("--------------------------------------------")

    #---------------------------- EVALUATE NETWORK ----------------------------

    ntest = ntracks-val_idx+1
    test_loss = track_index = 0
    test_results = np.zeros((ntest,max_len,2))
    with th.no_grad():
        model.eval()
        for ibatch, data in enumerate(test_loader):
            batch, test_labels = data
            pred = model(batch)
            pred_lt = loss(pred, test_labels)
            test_loss += pred_lt.item()

            test_labels[batch[:,:,0] == 0] = -1 #mark empty hits as -1

            test_results[track_index:track_index+batch.shape[0],:,:1] = pred
            test_results[track_index:track_index+batch.shape[0],:,1:] = test_labels
            track_index += batch.shape[0]

    test_results[:,:,0] = test_results[:,:,0] > class_threshold #round predictions based on chosen threshold

    test_loss = test_loss/ntest
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("Test loss: {}".format(test_loss), flush=True)

    eval_array = np.zeros((ntest, 2))
    for itrack in range(ntest):
        track_len = np.sum(test_results[itrack,:,1] != -1)
        correct = np.sum(test_results[itrack,:,0] == test_results[itrack,:,1])
        eval_array[itrack] = [track_len, correct]

    #plot number of correct hits vs number of total hits
    bin_edges = np.arange(-0.5,max_len+1.5,1)
    fig1 = plt.figure()
    plt.hist2d(eval_array[:,0], eval_array[:,1], bins=[bin_edges, bin_edges])
    #plt.scatter(eval_array[:,0], eval_array[:,1])
    plt.xlabel("Total hits")
    plt.ylabel("Correct hits")
    plt.savefig(outdir+str(runnumber)+"/correct_hits.png")

    #plot loss
    fig2 = plt.figure()
    plt.ioff()
    plt.plot(range(nepochs), train_loss_array, label="Training")
    plt.plot(range(nepochs), val_loss_array, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig(outdir+str(runnumber)+"/lossplot.png")

    tracks_file.close()


if __name__ == '__main__':
    main(sys.argv)
