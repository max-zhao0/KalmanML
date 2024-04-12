import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.stats as stat
import utils

def load_tracks(path, event_id):
    event_id = str(event_id)
    tracks = []
    tracks_file = h5py.File(path, "r")
    dataset = tracks_file[str(event_id) + "/tracks"]
    for track in dataset:
        tracks.append(np.array(track[track[:,-1] > 0]))
    tracks_file.close()
    return tracks

def calc_purity(tracks, fatras_particles):
    purity = np.empty((len(tracks), 4))
    particles = set()
    for i, track in enumerate(tracks):
        mode_result = stat.mode(track[:,-1], axis=None)
        particles.add(mode_result.mode[0])
        purity[i,0] = mode_result.count / track.shape[0]
        purity[i,1] = mode_result.mode[0]
        matched_particle = fatras_particles[fatras_particles[:,0] == purity[i,1]][0]
        purity[i,2] = matched_particle[1]
        purity[i,3] = matched_particle[3]

    return purity, particles

def calc_efficiency(fatras_particles, rec_particles):
    particles = np.empty((fatras_particles.shape[0], fatras_particles.shape[1]+1))
    particles[:,1:] = fatras_particles
    for i, part in enumerate(particles):
        particles[i,0] = int(part[1] in rec_particles)

    return particles

def plot_efficiency(particles, outdir, thresholds):
    plt.figure(figsize=(8,6))

    for i, parts in enumerate(particles):
        etas_edges, spacing = np.linspace(-3, 3, num=40, retstep=True)
        etas = etas_edges[:-1]
        totals = np.zeros(etas.shape)
        matches = np.zeros(etas.shape)
        for part in parts:
            index = np.argwhere(np.logical_and(etas < part[2], part[2] <= etas+spacing))[0,0]
            matches[index] += part[0]
            totals[index] += 1

        totals[totals == 0] = 1
        plt.stairs(100*matches/totals, etas_edges, label="{}%".format(thresholds[i]))

    plt.xlim(-3, 3)
    plt.title("Efficiency by policy threshold")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Efficiency (%)")
    plt.legend()
    plt.savefig(outdir + "eff_eta.png")

    plt.figure(figsize=(8,6))
    for i, parts in enumerate(particles):
        log_pt = np.log10(parts[:,4])
        log_pt_edges, spacing = np.linspace(min(log_pt), max(log_pt), num=40, retstep=True)
        log_pts = log_pt_edges[:-1]
        totals = np.zeros(log_pts.shape)
        matches = np.zeros(log_pts.shape)
        for j, part in enumerate(parts):
            index = np.argwhere(np.logical_and(log_pts <= log_pt[j], log_pt[j] <= log_pts+spacing))[0,0]
            matches[index] += part[0]
            totals[index] += 1

        totals[totals == 0] = 1
        plt.stairs(100*matches/totals, log_pt_edges, label="{}%".format(thresholds[i]))

    plt.xlim(min(log_pt), max(log_pt))
    plt.title("Efficiency by policy threshold")
    plt.xlabel(r"$\log_{10} p_T$")
    plt.ylabel("Efficiency (%)")
    plt.legend()
    plt.savefig(outdir + "eff_pt.png")

def plot_purity(purities, outdir, thresholds):
    plt.figure(figsize=(8,6))

    for i, purity in enumerate(purities):
        etas_edges, spacing = np.linspace(-3, 3, num=40, retstep=True)
        etas = etas_edges[:-1]
        totals = np.zeros(etas.shape)
        matches = np.zeros(etas.shape)
        for rec_part in purity:
            # print(etas+spacing)
            # print(rec_part)
            # print(np.logical_and(etas <= rec_part[2], rec_part[2] <= etas+spacing))
            # print(np.argwhere(np.logical_and(etas <= rec_part[2], rec_part[2] <= etas+spacing)))
            index = np.argwhere(np.logical_and(etas <= rec_part[2], rec_part[2] <= etas+spacing))[0,0]
            matches[index] += rec_part[0]
            totals[index] += 1

        totals[totals == 0] = 1
        plt.stairs(100*matches/totals, etas_edges, label="{}%".format(thresholds[i]))

    plt.xlim(etas_edges[0], etas_edges[-1])
    plt.title("Purity by policy threshold")
    plt.xlabel(r"$\eta$")
    plt.ylabel("Purity (%)")
    plt.legend()
    plt.savefig(outdir + "purity_eta.png")

    plt.figure(figsize=(8,6))

    for i, purity in enumerate(purities):
        log_pt = np.log10(purity[:,3])
        log_pt_edges, spacing = np.linspace(min(log_pt), max(log_pt), num=40, retstep=True)
        log_pts = log_pt_edges[:-1]
        totals = np.zeros(log_pts.shape)
        matches = np.zeros(log_pts.shape)
        for j, rec_part in enumerate(purity):
            index = np.argwhere(np.logical_and(log_pts <= log_pt[j], log_pt[j] <= log_pts+spacing))[0,0]
            matches[index] += rec_part[0]
            totals[index] += 1

        totals[totals == 0] = 1
        plt.stairs(100*matches/totals, log_pt_edges, label="{}%".format(thresholds[i]))

    plt.xlim(log_pt_edges[0], log_pt_edges[-1])
    plt.title("Purity by policy threshold")
    plt.xlabel(r"$\log_{10} p_T$")
    plt.ylabel("Purity (%)")
    plt.legend()
    plt.savefig(outdir + "purity_pt.png")

def main(argv):
    indir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/tracks/policy_tracker/"
    trackml_dir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/"
    outdir = "./"
    thresholds = [50, 90, 99]

    purity = []
    particles = []
    for thresh in thresholds:
        tracks_path = indir + "tracks{}.hdf5".format(thresh)
        purity_lst = []
        part_lst = []
        for event_id in range(80, 100):
            fatras_path = trackml_dir + "events{}/ttbar200_10/processed/particles.hdf5".format(int(event_id/10))
            fatras_file = h5py.File(fatras_path, "r")
            fatras_particles = fatras_file[str(event_id % 10) + "/particles"]

            event_tracks = load_tracks(tracks_path, event_id)
            event_purity, rec_particles = calc_purity(event_tracks, fatras_particles)
            #event_part = calc_efficiency(fatras_particles, rec_particles)
            purity_lst.append(event_purity)
            #part_lst.append(event_part)

            fatras_file.close()

        purity.append(np.concatenate(purity_lst, axis=0))
        #particles.append(np.concatenate(part_lst, axis=0))

    #plot_efficiency(particles, outdir, thresholds)
    plot_purity(purity, outdir, thresholds)

    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))