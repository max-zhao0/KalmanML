#!/usr/bin/env python

import os, sys, h5py, argparse
from tqdm import tqdm
import numpy as np
from ROOT import gROOT, TFile


def main(argv):
    gROOT.SetBatch(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help = 'Path containing data generated with ACTS (with trackstates_ckf.root file)')
    parser.add_argument('--out_dir', type=str, help = 'Path containing pre-processed data (with hits.hdf5 file)')
    parser.add_argument('--hit_bounds', nargs=2, type=int, default=[3, 30], help = 'Bounds for number of hits in track')
    parser.add_argument('-o', '--overwrite', action='store_true', help = 'Overwrite existing output file')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir+"hits.hdf5"):
        raise FileNotFoundError("File hits.hdf5 not found in output directory (please make sure to run import_hits.py first)")
    else:
        hits_file = h5py.File(args.out_dir+"hits.hdf5", "r")
    if args.overwrite or not os.path.exists(args.out_dir+"hits.hdf5"):
        outfile = h5py.File(args.out_dir+"tracks.hdf5","w")
    else:
        raise FileExistsError("Output file tracks.hdf5 already exists. Use -o to overwrite")
    
    tracks_file = TFile(args.in_dir+"trackstates_ckf.root")
    tracks_tree = tracks_file.Get("trackstates")
 
    nentries = tracks_tree.GetEntries()

    track_meas_dtypes = np.dtype([('volume_id', '<i2'), ('layer_id', '<i2'), ('module_id', '<i2'), ('rec_loc0', '<f4'), ('rec_loc1', '<f4'), ('channel_value', '<f4'), ('clus_size', '<f4'), ('dim_hit', '<i2'), ('g_x', '<f4'), ('g_y', '<f4'), ('g_z', '<f4'), ('res_x', '<f4'), ('res_y', '<f4'), ('err_x', '<f4'), ('err_y', '<f4')])
    track_truth_dtypes = np.dtype([('NN_label', '<i2'), ('particle_id', '<f4'), ('t_x', '<f4'), ('t_y', '<f4'), ('t_z', '<f4')])

    track_meas = np.zeros((nentries, args.hit_bounds[1]), dtype=track_meas_dtypes)
    track_truth = np.zeros((nentries, args.hit_bounds[1]), dtype=track_truth_dtypes)

    percents_match = []
    bad_tracks = [] #index of tracks to remove from final sample

    for ientry, track_entry in enumerate(tqdm(tracks_tree, total=nentries)):
        nhits = track_entry.nMeasurements # number of measurements in track
        nstates = track_entry.nStates # number of CKF states in track
        event_id = track_entry.event_nr

        good_hits = ~np.isnan(track_entry.t_x) #skip CKF states without corresponding measurement

        volume_id = np.array(track_entry.volume_id)[good_hits]
        layer_id = np.array(track_entry.layer_id)[good_hits]
        module_id = np.array(track_entry.module_id)[good_hits]

        true_x = np.array(track_entry.t_x)[good_hits] #true x coordinate of hit
        true_y = np.array(track_entry.t_y)[good_hits] #true y coordinate of hit
        true_z = np.array(track_entry.t_z)[good_hits] #true z coordinate of hit

        meas_dim =  np.array(track_entry.dim_hit) #dimension of measurement
        #meas_lx =  np.array(track_entry.l_x_hit)[good_hits] #local x coordinate of hit
        #meas_ly =  np.array(track_entry.l_y_hit)[good_hits] #local y coordinate of hit
        meas_x =  np.array(track_entry.g_x_hit)[good_hits] #global x coordinate of hit
        meas_y =  np.array(track_entry.g_y_hit)[good_hits] #global y coordinate of hit
        meas_z =  np.array(track_entry.g_z_hit)[good_hits] #global z coordinate of hit
        meas_res_x =  np.array(track_entry.res_x_hit) #hit residual in x
        meas_res_y =  np.array(track_entry.res_y_hit) #hit residual in y
        meas_err_x = np.array(track_entry.err_x_hit) #hit error in x
        meas_err_y = np.array(track_entry.err_y_hit) #hit error in y
        #meas_pull_x = np.array(track_entry.pull_x_hit) #hit pull in x
        #meas_pull_y = np.array(track_entry.pull_y_hit) #hit pull in y

        if nhits > args.hit_bounds[0]:
            for i in range(nhits):
                rel_hits = hits_file[str(event_id)+"/"+str(volume_id[i])]["hits"]

                hitmatch = np.logical_and.reduce((rel_hits['layer_id'] == layer_id[i], rel_hits['surface_id'] == module_id[i], np.abs(rel_hits['true_x'] - true_x[i]) < 10e-4, np.abs(rel_hits['true_y'] - true_y[i]) < 10e-4, np.abs(rel_hits['true_z'] - true_z[i]) < 10e-4))

                #check if hit match was found, if not add to bad tracks
                if np.where(hitmatch)[0].size == 0:
                    bad_tracks.append(ientry)
                    break
                else:
                    hitindex = np.where(hitmatch)[0][0]
                    
                    try:
                        track_truth[ientry, i] = (0, rel_hits['particle_id'][hitindex], true_x[i], true_y[i], true_z[i])
                        track_meas[ientry, i] = (volume_id[i], layer_id[i], module_id[i], rel_hits['rec_loc0'][hitindex], rel_hits['rec_loc0'][hitindex], rel_hits['channel_value'][hitindex], rel_hits['clus_size'][hitindex], meas_dim[i], meas_x[i], meas_y[i], meas_z[i], meas_res_x[i], meas_res_y[i], meas_err_x[i], meas_err_y[i])
                    except IndexError:
                        print("IndexError:", i)

            if ientry not in bad_tracks:
                valid_hits = track_truth['particle_id'][ientry] > 0
                track_truth[ientry][valid_hits] = np.flip(track_truth[ientry][valid_hits]) #flip hits in track to start from beamline
                track_meas[ientry][valid_hits] = np.flip(track_meas[ientry][valid_hits])

                track_truth['NN_label'][ientry] = np.logical_and(track_truth['particle_id'][ientry] == track_truth['particle_id'][ientry,0], track_truth['particle_id'][ientry] > 0) #calculate NN labels by checking which hits are from same particle as seed

                track_candidate = track_truth['particle_id'][ientry][valid_hits] #isolate track candidate (useful for plotting)
                try:
                    percent_from_seed_particle = track_candidate[track_candidate == track_candidate[0]].shape[0]/track_candidate.shape[0] #calculate percentage of hits in track that are from same particle as seed
                    percents_match.append(percent_from_seed_particle)
                except IndexError:
                    print("Empty track")
                    bad_tracks.append(ientry)
            
            # if ientry % 1000 == 0:
            #     plt.figure(figsize=(8,6))
            #     plt.hist(percents_match, bins=40, label="{:.3f}%".format(np.mean(percents_match)*100))
            #     plt.title("CKF match percentage with first hit")
            #     plt.legend()
            #     plt.savefig(outdir + "ckf_match_rate.png")
            #     plt.close()

    hits_file.close()

    #remove bad tracks
    track_meas = np.delete(track_meas, bad_tracks, axis=0)
    track_truth = np.delete(track_truth, bad_tracks, axis=0)
    print("\nRemoved {} bad tracks".format(len(bad_tracks)))

    track_meas = track_meas[track_truth['particle_id'][:,0] > 0]
    track_truth = track_truth[track_truth['particle_id'][:,0] > 0]
    # plot_match_distribution(track_truth, "./pre_match_dist.png")
    track_truth, unique_indices = np.unique(track_truth,return_index=True,axis=0)   
    # plot_match_distribution(track_truth, "./unique_match_dist.png")
    track_meas = track_meas[unique_indices]

    shuffle_array = np.random.permutation(track_truth.shape[0])
    track_truth = track_truth[shuffle_array]
    track_meas = track_meas[shuffle_array]

    #calculate what percent tracks are from unique seeds
    percent_from_unique_seed = np.unique(track_truth['particle_id'][:,0]).shape[0]/track_truth['particle_id'].shape[0]

    # plot_candidates(track_truth,"xy",10000)

    group = outfile.create_group("tracks")
    group.create_dataset("measurements",data=track_meas)
    group.create_dataset("truth",data=track_truth)

    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
