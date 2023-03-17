import sys
import numpy as np
import utils

def make_locate_layer(hit_locator):
    Z1_BOUND = 3000
    Z0_BOUND = 1800
    BARRELS = {8, 13, 17}
    
    t_range, u_range, layer_dict = hit_locator.get_detector_spec()
    assert t_range[7][0] <= u_coord[8, 2]
    R_range0 = t_range[7]
    R_range1 = t_range[12]
    R_range2 = t_range[16]
    
    def locate_layer(point):
        """
        Finding of nearest layer
        """
        if abs(point[2]) > Z1_BOUND:
            return 2, None, None, None
        
        r = np.sqrt(point[0]**2 + point[1]**2)
        if r < R_range[0] or R_range0[1] < r < R_range1[0] or R_range1[1] < r < R_range2[0]:
            return 1, None, None, None
        elif R_range0[0] <= r <= R_range0[1]:
            if abs(point[2]) > Z0_BOUND:
                return 1, None, None, None
            elif point[2] < t_range[8][0]:
                volume = 7
            elif point[2] > t_range[8][1]:
                volume = 9
            else:
                volume = 8
        elif R_range1[0] <= r <= R_range1[1]:
            if point[2] < t_range[13][0]:
                volume = 12
            if point[2] > t_range[13][1]:
                volume = 14
            else:
                volume = 13
        elif R_range2[0] <= r <= R_range2[1]:
            if point[2] < t_range[17][0]:
                volume = 16
            elif point[2] > t_range[17][1]:
                volume = 18
            else:
                volume = 13
        else:
            return 2, None, None, None
        
        min_dist = np.inf
        phi = np.arctan2(point[1], point[0])
        if volume in BARRELS:
            for lay in layer_dict[volume]:
                dist = abs(u_coord[volume, lay] - r)
                if dist < min_dist:
                    min_dist = dist
                    layer = lay
            projection = (phi, point[2])
        else:
            for lay in layer_dict[volume]:
                dist = abs(u_coord[volume, lay] - point[2])
                if dist < min_dist:
                    min_dist = dist
                    layer = lay
            projection = (phi, r)

        return 0, volume, layer, min_dist, projection
    
    return locate_layer

def propagate(points, stepsize, radius, hit_locator, locate_layer, bmap):
    avg_pos = np.mean(points, axis=0)
    stepper = utils.helix_stepper(points, bmap.get(avg_pos), stepsize)
    
    near_next_layer = False
    near_start_layer = True
    while not near_next_layer:
        curr_pos = next(stepper)

        exit_code, volume, closest_layer, dist, projection = locate_layer(curr_pos)
        if exit_code == 1:
            continue
        elif exit_code == 2:
            return None
        
        if near_start_layer and dist >= 2*stepsize:
            near_start_layer = False
        elif not near_start_layer and dist < 2*stepsize:
            near_next_layer = True
    
    return hit_locator.get_hits_around(volume, closest_layer, projection, radius)

def truth_seeds(locator):
    first_layer = locator.get_layer_hits(8, 2)
    second_layers = [locator.get_layer_hits(8, 4), locator.get_layer_hits(7, 14), locator.get_layer_hits(9, 2)]
    
    hit_pairs = []
    for first_hit in first_layer:
        for second_layer in second_layers:
            matches = second_layer[second_layer[:,-1] == first_hit[-1]]
            for second_hit in matches:
                hit_pairs.append(np.array([first_hit, second_hit]))
    
    return np.array(hit_pairs)

def generate_triplets(hits_path, detector_path, event_id):
    locator_resolution = 5
    stepsize = 5
    radius = 50
    
    locator = utils.HitLocator(locator_resolution, detector_path)
    locate_layer = make_locate_layer(locator)
    bmap = utils.BFieldMap()
    seeds = truth_seeds(locator)
    
    triplets = []
    
    for seed in seeds:
        prev_triplet = np.array([np.zeros(10), seed[0], seed[1]])
        
        track_incomplete = True
        while track_incomplete:
            new_hits = propagate(prev_triplet[:,6:9], stepsize, radius, locator, locate_layer, bmap)
            if new_hits is None:
                track_incomplete = False
            else:
                matches = new_hits[new_hits[:,-1] == prev_triplet[-1,-1]]
                if not matches.shape[0]:
                    track_incomplete = False
                else:
                    prev_triplet = np.append(prev_triplet[1:], [matches[0]])
                    triplets.append(prev_triplet)
                    
    return triplets
  
def main(argv):
    hits_path = None
    detector_path = None
    outdir = None
    nevents = 10
    
    triplets = []
    for event_id in range(nevents):
        triplets += generate_triplets(hits_path, detector_path, event_id)
    
    np.savetxt(outdir + "triplets.csv", np.array(triplets), delimiter=",")
    
    return 0
  
if __name__ == "__main__":
    print("Finished with exit code:", main(sys.argv))
