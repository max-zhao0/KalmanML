import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

def main(argv):
    metrics_path = "/global/homes/m/max_zhao/bin/policy_winter_metrics.csv"
    quadruplets_path = "/pscratch/sd/m/max_zhao/policy/quadruplets_intersections.hdf5"
    # intersections_path = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/ttbar200_100/processed/intersections.hdf5"
    test_events = range(80, 100)
    max_radius = 50

    quadruplets_file = h5py.File(quadruplets_path, "r")
    #intersections_file = h5py.File(intersections_path, "r")

    false_positive_rate, false_negative_rate = np.loadtxt(metrics_path, delimiter=",", unpack=True)
    best_epoch = np.argmin(false_positive_rate + false_negative_rate)
    print(best_epoch)
    best_fpr = false_positive_rate[best_epoch]
    best_fnr = false_negative_rate[best_epoch]

    positive_distribution = np.zeros(max_radius)
    negative_distribution = np.zeros(max_radius)
    for event in test_events:
        quadruplets = quadruplets_file["40/" + str(event) + "/quadruplets"][:,:4,:]
        labels = quadruplets_file["40/" + str(event) + "/labels"]
        intersections = quadruplets_file["40/" + str(event) + "/quadruplets"][:,4,10:13]

        event_distances = np.sqrt(np.sum((quadruplets[:,-1,10:13] - intersections[:])**2, axis=1))
        for i, dist in enumerate(event_distances):
            if np.isnan(dist):
                continue
            if intersections[i,0] != 0 and int(dist) < max_radius:
                if labels[i]:
                    positive_distribution[int(dist)] += 1
                else:
                    negative_distribution[int(dist)] += 1

    quadruplets_file.close()
    #intersections_file.close()

    trivial_metrics = np.zeros((max_radius-1,2))
    for rad in range(1, max_radius):
        trivial_fpr = np.sum(negative_distribution[:rad]) / np.sum(negative_distribution)
        trivial_fnr = np.sum(positive_distribution[rad:]) / np.sum(positive_distribution)
        trivial_metrics[rad-1] = np.array([trivial_fpr, trivial_fnr])

    plt.figure(figsize=(8,6))

    plt.plot(np.arange(1,max_radius), trivial_metrics[:,0], label="Trivial FPR", color="blue")
    plt.plot(np.arange(1,max_radius), trivial_metrics[:,1], label="Trivial FNR", color="orange")
    plt.hlines(best_fpr, 0, 50, label="NN FPR", color="blue")
    plt.hlines(best_fnr, 0, 50, label="NN FNR", color="orange")
    
    plt.title("Trivial estimator compared to NN")
    plt.xlabel("Threshold radius (mm)")
    plt.ylabel("Rate")

    plt.legend()
    plt.savefig("trivial_comparison.png")

    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))