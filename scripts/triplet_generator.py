import sys
import numpy as np
import utils
import time

BARRELS = {8, 13, 17}

def make_locate_layer(hit_locator):
    Z1_BOUND = 3000
    Z0_BOUND = 1800
    
    t_range, u_coord, layer_dict = hit_locator.get_detector_spec()
    assert t_range[7][0] <= u_coord[8, 2]
    R_range0 = t_range[7]
    R_range1 = t_range[12]
    R_range2 = t_range[16]
    
    def locate_layer(point):
        """
        Finding of nearest layer
        """
        if abs(point[2]) > Z1_BOUND:
            return 2, None, None, None, None
        
        r = np.sqrt(point[0]**2 + point[1]**2)
        if r < R_range0[0] or R_range0[1] < r < R_range1[0] or R_range1[1] < r < R_range2[0]:
            return 1, None, None, None, None
        elif R_range0[0] <= r <= R_range0[1]:
            if abs(point[2]) > Z0_BOUND:
                return 1, None, None, None, None
            elif point[2] < t_range[8][0]:
                volume = 7
            elif point[2] > t_range[8][1]:
                volume = 9
            else:
                volume = 8
        elif R_range1[0] <= r <= R_range1[1]:
            if point[2] < t_range[13][0]:
                volume = 12
            elif point[2] > t_range[13][1]:
                volume = 14
            else:
                volume = 13
        elif R_range2[0] <= r <= R_range2[1]:
            if point[2] < t_range[17][0]:
                volume = 16
            elif point[2] > t_range[17][1]:
                volume = 18
            else:
                volume = 17
        else:
            return 2, None, None, None, None
        
        min_dist = np.inf
        raw_phi = np.arctan2(point[1], point[0])
        phi = raw_phi if raw_phi >= 0 else 2*np.pi + raw_phi
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

def propagate(points, stepsize, radius, hit_locator, locate_layer, bmap, helices=None):
    avg_pos = np.mean(points, axis=0)
    stepper = utils.helix_stepper(points, bmap.get(avg_pos), stepsize, helices)

    _, curr_vol, curr_lay, _, _ = locate_layer(points[-1])
    _, u_coord, _ = hit_locator.get_detector_spec()

    near_next_layer = False
    while not near_next_layer:
        curr_pos = next(stepper)

        exit_code, volume, closest_layer, dist, projection = locate_layer(curr_pos)
        if exit_code == 1:
            continue
        elif exit_code == 2:
            return None, None
        
        different_layer = volume != curr_vol or closest_layer != curr_lay
        near_next_layer = different_layer and dist < 2*stepsize
    
    if volume in BARRELS:
        r = u_coord[volume,closest_layer]
        phi, z = projection
    else:
        z = u_coord[volume,closest_layer]
        phi, r = projection
    proj_3d = np.array([r*np.cos(phi), r*np.sin(phi), z])

    return hit_locator.get_hits_around(volume, closest_layer, projection, radius), proj_3d

def truth_seeds_DEPRECATED(locator):
    first_layer = locator.get_layer_hits(8, 2)
    second_layers = [locator.get_layer_hits(8, 4), locator.get_layer_hits(7, 14), locator.get_layer_hits(9, 2)]
    
    hit_pairs = []
    for first_hit in first_layer:
        for second_layer in second_layers:
            matches = second_layer[second_layer[:,-1] == first_hit[-1]]
            for second_hit in matches:
                hit_pairs.append(np.array([first_hit, second_hit]))
    
    return np.array(hit_pairs)

def truth_seeds(locator, pcut=None):
    layer_graph = {
        (8, 2) : [(8,4), (7,14), (9,2)],
        (7, 14) : [(7,12)],
        (8, 4) : [(8,6), (7,14), (9,2)],
        (9,2) : [(9,4)]
    }
    first_layer = locator.get_layer_hits(8, 2)
    seeds = []
    
    for first_hit in first_layer:
        particle_id = first_hit[-1]
        seed = [first_hit]
        seed_layers = [(8, 2)]

        while len(seed) < 3:
            next_layer_tups = layer_graph[seed_layers[-1]]

            found = False
            for layer_tup in next_layer_tups:
                next_layer = locator.get_layer_hits(*layer_tup)
                matches = next_layer[next_layer[:,-1] == particle_id]
                if matches.shape[0] > 0:
                    found = True
                    seed.append(matches[0])
                    seed_layers.append(layer_tup)
                    break

            if not found:
                break

        if len(seed) == 3:
            seeds.append(seed)

    return np.array(seeds)

def generate_triplets(hits_path, detector_path, event_id, save_helices=False):
    locator_resolution = 5
    stepsize = 5
    radius = 150
    
    locator = utils.HitLocator(locator_resolution, detector_path)
    locator.load_hits(hits_path, event_id)

    locate_layer = make_locate_layer(locator)
    bmap = utils.BFieldMap()
    seeds = truth_seeds(locator)

    triplets = []
    labels = []
    finding_stats = []
    if save_helices:
        helices = []
    
    prop_break = 0
    
    for seed in seeds:
        # Special treatmenet for seed
        if True:
            exit_code, volume, closest_layer, dist, projection = locate_layer(seed[-1,6:9])
            assert exit_code == 0
            new_hits = locator.get_hits_around(volume, closest_layer, projection, radius)
            match_ind = new_hits[:,-1] == seed[-1,-1]

            for hit in new_hits:
                triplets.append(np.append(seed[:-1], np.array([hit]), 0))
            labels.append(np.int32(match_ind))
            finding_stats.append(np.append(seed[-1,6:9], np.array([1])))

        prev_triplet = seed
        track_incomplete = True
        while track_incomplete:
            new_hits, predicted_pos = propagate(prev_triplet[:,6:9], stepsize, radius, locator, locate_layer, bmap, helices)

            if new_hits is None or len(new_hits) == 0:
                # Terminate track if no new hits are found
                track_incomplete = False
            else:
                match_ind = new_hits[:,-1] == prev_triplet[-1,-1]
                #labels = np.concatenate((labels, np.int32(match_ind)))
                labels.append(np.int32(match_ind))
                for hit in new_hits:
                    triplets.append(np.append(prev_triplet[1:], np.array([hit]), 0))

                matches = new_hits[match_ind]
                not_matches = new_hits[~match_ind]
                
                if matches.shape[0] == 0:
                    track_incomplete = False
                else:
                    if matches[0,6:9] in prev_triplet[:,6:9]:
                        prop_break += 1
                        break
                    prev_triplet = np.append(prev_triplet[1:], [matches[0]], axis=0)
            
            if predicted_pos is not None:
                finding_stats.append(np.append(predicted_pos, np.array([np.int32(track_incomplete)])))

    print("Event: " + str(event_id) + " Break rate: " + str(prop_break / seeds.shape[0]))
    labels = np.concatenate(labels)
    assert len(labels) == len(triplets)

    if save_helices:
        return triplets, labels, finding_stats, helices
    else:
        return triplets, labels, finding_stats
  
def main(argv):
    hits_path = "/global/cfs/cdirs/atlas/jmw464/mlkf_data/python/ttbar60_10/processed/hits.hdf5"
    detector_path = "/global/homes/m/max_zhao/mlkf/trackml/data/detectors.csv"
    outdir = "/global/homes/m/max_zhao/mlkf/trackml/data/triplets/"
    nevents = 10
    
    triplets = []
    labels = []
    finding_stats = []
    helices = []
    for event_id in range(nevents):
        event_triplets, event_labels, event_finding_stats, event_helices = generate_triplets(hits_path, detector_path, event_id, True)

        triplets += event_triplets
        labels = np.concatenate((labels, event_labels))
        finding_stats += event_finding_stats
        helices += event_helices

    finding_stats = np.array(finding_stats)
    helices = np.array(helices)

    print("Total finding rate:", np.sum(finding_stats[:,-1]) / finding_stats.shape[0])
    np.savetxt(outdir + "finding_stats150.csv", finding_stats, delimiter=",")
    np.savetxt(outdir + "helices.csv", helices, delimiter=",")

    print(sum(labels), len(labels))
    return 0
  
if __name__ == "__main__":
    print("\nFinished with exit code:", main(sys.argv))
