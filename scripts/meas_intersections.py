import sys
import numpy as np
import h5py
import time
import warnings
import helices
import utils

warnings.filterwarnings("error", category=RuntimeWarning)

def main(argv):
    inpath = "/pscratch/sd/m/max_zhao/policy/quadruplets.hdf5"
    outpath = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/ttbar200_100/processed/intersections.hdf5"
    detector_path = "/global/homes/m/max_zhao/mlkf/trackml/data/detectors.csv"
    ignore_path = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/ttbar200_100/processed/intersection_breaks.csv"

    infile = h5py.File(inpath, "r")
    outfile = h5py.File(outpath, "w")
    bmap = utils.BFieldMap()
    geometry = utils.Geometry(detector_path, bmap)

    ignore_points = []
    for event_id in infile["40"].keys():
        start_time = time.time()

        quadruplets = infile["40/" + event_id + "/quadruplets"]
        event_intersections = np.zeros((quadruplets.shape[0], 3))
        
        cached_triplet = np.zeros((3, 13))
        cached_intersection = None

        used_cache = 0
        for ientry, quad in enumerate(quadruplets):
            if np.all(quad[:3] == cached_triplet):
                used_cache += 1
                event_intersections[ientry] = cached_intersection
            else:
                try:
                    helix = helices.Helix()
                    helix.solve(*quad[:3,10:13], bmap.get(quad[2,10:13]))
                    intersection, _, _ = helices.find_helix_intersection(helix, geometry, 5)
                    event_intersections[ientry] = intersection
                    cached_intersection = intersection
                    cached_triplet = quad[:3]
                except RuntimeWarning:
                    print("Runtime error, Event: {}, Index: {}".format(event_id, ientry))
                    ignore_points.append([event_id, ientry])
                
            # print(np.sqrt(np.sum((event_intersections[ientry] - quad[3,10:13])**2)))

        group = outfile.create_group("40/" + event_id)
        group.create_dataset("intersections", data=event_intersections)

        print("Finished event: " + event_id + " Time:", time.time()-start_time)
        print("Used caches: {}/{}".format(used_cache, quadruplets.shape[0]))

    infile.close()
    outfile.close()

    print("Ignore points:", len(ignore_points))
    np.savetxt(ignore_path, np.array(ignore_points), delimiter=",")
    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))