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

    track_meas_dtypes = np.dtype([('volume_id', '<i2'), ('layer_id', '<i2'), ('module_id', '<i2'), ('rec_loc0', '<i2'), ('rec_loc1', '<f4'), ('channel_value', '<f4'), ('clus_size', '<f4')])
    track_truth_dtypes = np.dtype([('NN_label', '<i4'), ('particle_id', '<f4'), ('t_x', '<f4'), ('t_y', '<f4'), ('t_z', '<i2')])

    track_meas = np.zeros((nentries, args.hit_bounds[1]), dtype=track_meas_dtypes)
    track_truth = np.zeros((nentries, args.hit_bounds[1]), dtype=track_truth_dtypes)

    percents_match = []
    bad_tracks = [] #index of tracks to remove from final sample

    for ientry, track_entry in enumerate(tqdm(tracks_tree, total=nentries)):
        nhits = track_entry.nMeasurements
        event_id = track_entry.event_nr
        if nhits > args.hit_bounds[0]:
            for i in range(nhits):
                volume_id = track_entry.volume_id[i]
                layer_id = track_entry.layer_id[i]
                module_id = track_entry.module_id[i]

                rel_hits = hits_file[str(event_id)+"/"+str(volume_id)]["hits"]
                true_x = track_entry.t_x[i]
                true_y = track_entry.t_y[i]
                true_z = track_entry.t_z[i]

                hitmatch = np.logical_and.reduce((rel_hits['layer_id'] == layer_id, rel_hits['surface_id'] == module_id, np.abs(rel_hits['true_x'] - true_x) < 10e-4, np.abs(rel_hits['true_y'] - true_y) < 10e-4, np.abs(rel_hits['true_z'] - true_z) < 10e-4))

                #check if hit match was found, if not add to bad tracks
                if np.where(hitmatch)[0].size == 0:
                    bad_tracks.append(ientry)
                    break
                else:
                    hitindex = np.where(hitmatch)[0][0]
                    
                    try:
                        track_truth[ientry, nhits-i-1] = (0, rel_hits['particle_id'][hitindex], true_x, true_y, true_z)
                        track_meas[ientry, nhits-i-1] = (volume_id, layer_id, module_id, rel_hits['rec_loc0'][hitindex], rel_hits['rec_loc0'][hitindex], rel_hits['channel_value'][hitindex], rel_hits['clus_size'][hitindex])
                    except IndexError:
                        print("IndexError:", nhits-i-1)

            if ientry not in bad_tracks:
                track_truth['NN_label'][ientry] = np.logical_and(track_truth['particle_id'][ientry] == track_truth['particle_id'][ientry,0], track_truth['particle_id'][ientry] > 0) #calculate NN labels by checking which hits are from same particle as seed
                track_candidate = track_truth['particle_id'][ientry][track_truth['particle_id'][ientry] > 0] #isolate track candidate (useful for plotting)
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
    percent_from_unique_seed = np.unique(track_truth['particle_id'][:,0]).shape[0]/track_truth.shape[0]

    # plot_candidates(track_truth,"xy",10000)

    group = outfile.create_group("tracks")
    group.create_dataset("measurements",data=track_meas)
    group.create_dataset("truth",data=track_truth)

    hits_file.close()
    outfile.close()


if __name__ == '__main__':
    main(sys.argv)
