import numpy as np
import h5py

class HitLocator:
    # Volume ids in the detector
    volume_lst = np.array([7, 8, 9, 12, 13, 14, 16, 17, 18])
    # Volumes that are barrel shaped
    barrel_set = {8, 13, 17}
    
    hit_map = {}
    t_range = {}

    def __init__(self, resolution, detector_path):
        """
        Initialize the data structure with empty cells using detector geometry file.
        Detector geometry file can be found at https://www.kaggle.com/competitions/trackml-particle-identification/data.
        ---
        resolution      : float     : width of cells
        detector_path   : String    : path to detector csv file.
        """
        volume_id, layer_id, module_id, cx, cy, cz, hv = np.loadtxt(detector_path, delimiter=",", skiprows=1, usecols=[0,1,2,3,4,5,18], unpack=True)
        
        for volume in self.volume_lst:
            lay_id_vol = layer_id[volume_id == volume]
            cx_vol = cx[volume_id == volume]
            cy_vol = cy[volume_id == volume]
            cz_vol = cz[volume_id == volume]
            hv_vol = hv[volume_id == volume]
            max_hv = max(hv_vol)

            vol_map = {}
            if volume in self.barrel_set:
                min_z = min(cz_vol) - max_hv
                max_z = max(cz_vol) + max_hv
                self.t_range[volume] = (min_z, max_z)
                
                for layer in set(lay_id_vol):
                    cx_lay = cx_vol[lay_id_vol == layer]
                    cy_lay = cy_vol[lay_id_vol == layer]
                    diameter = 2 * np.sqrt(cx_lay[0]**2 + cy_lay[0]**2)
                    
                    z_dim = round(np.ceil((max_z - min_z) / resolution))
                    phi_dim = round(np.ceil(np.pi * diameter / resolution))
                    vol_map[layer] = np.empty((phi_dim, z_dim), dtype=list)
            else:
                cr_vol = np.sqrt(cx_vol**2 + cy_vol**2)
                min_r = min(cr_vol) - max_hv
                max_r = max(cr_vol) + max_hv
                self.t_range[volume] = (min_r, max_r)
                
                r_dim = round(np.ceil((max_r - min_r) / resolution))
                phi_dim = round(np.ceil(np.pi * (max_r + min_r) / resolution))
                for layer in set(lay_id_vol):
                    vol_map[layer] = np.empty((phi_dim, r_dim), dtype=list)

            for layer in set(lay_id_vol):
                for row in vol_map[layer]:
                    for i in range(len(row)):
                        row[i] = []

            self.hit_map[volume] = vol_map

    def load_hits(self, hits_path, event_id):
        """
        Load hits into data structure from hits file
        ---
        hits_path   : String    : path to hits file
        event_id    : int       : event to store
        """
        event_id = str(event_id)
        f = h5py.File(hits_path, "r")

        for volume_id in f[event_id].keys():
            for layer_id in f[event_id + "/" + volume_id].keys():
                for module_id in f[event_id + "/" + volume_id + "/" + layer_id]:
                    data = f[event_id + "/" + volume_id + "/" + layer_id + "/" + module_id]["hits"]
                    volume = int(volume_id)
                    layer = int(layer_id)
                    lay_map = self.hit_map[volume][layer]
                    vol_range = self.t_range[volume]

                    # x, y, z = 4, 5, 6
                    for hit in data:
                        x, y, z = hit[4:7]

                        raw_phi = np.arctan2(x, y)
                        phi = raw_phi if raw_phi >= 0 else 2 * np.pi + raw_phi
                        phi_coord = round((phi / (2 * np.pi)) * (lay_map.shape[0] - 1))

                        if volume in self.barrel_set:
                            t_coord = round((lay_map.shape[1] - 1) * (z - vol_range[0]) / (vol_range[1] - vol_range[0]))
                        else:
                            r = np.sqrt(x**2 + y**2)
                            t_coord = round((lay_map.shape[1] - 1) * (r - vol_range[0]) / (vol_range[1] - vol_range[0]))
                        
                        lay_map[phi_coord, t_coord].append(hit)
        f.close()

    def get_near_hits(self, volume, layer, center, area):
        """
        Get all hits near some point on a layer
        ---
        volume  : int               : volume that contains the layer
        layer   : int               : layer number
        center  : (float, float)    : point around which to collect hits. Of the form (phi, t) where t = z if barrel volume and r if endcap
        area    : (float, float)    : range with which to collect hits. Essentially collect hits with coordinate in center +- area
        ---
        Returns:
        hits    : List              : list of hits
        """
        assert area[0] > 0 and area[1] > 0

        lay_map = self.hit_map[volume][layer]
        lay_range = self.t_range[volume]

        get_phi_coord = lambda phi: round((lay_map.shape[0] - 1) * phi / (2 * np.pi))
        get_t_coord = lambda t: round((lay_map.shape[1] - 1) * (t - lay_range[0]) / (lay_range[1] - lay_range[0]))

        start_phi = get_phi_coord(center[0] - area[0]) % lay_map.shape[0]
        end_phi = get_phi_coord(center[0] + center[0]) % lay_map.shape[0]
        start_t = max(get_t_coord(center[1] - area[1]), 0)
        end_t = min(get_t_coord(center[1] + area[1]), lay_map.shape[1] - 1)

        hits = []
        for t_coord in range(start_t, end_t + 1):
            phi_coord = start_phi
            while phi_coord != end_phi:
                hits += lay_map[phi_coord, t_coord]
                phi_coord = (phi_coord + 1) % lay_map.shape[0]

        return hits
 
if __name__ == "__main__":
    detector_path = "/global/homes/m/max_zhao/mlkf/trackml/data/detectors.csv"
    hits_path = "/global/homes/m/max_zhao/mlkf/trackml/data/hits.hdf5" 

    loc = HitLocator(10, detector_path)
    loc.load_hits(hits_path, 0)
    print("Finished initializing")
    hits = loc.get_near_hits(8, 8, (np.pi, 0), (np.pi / 7, 80))
    print(len(hits))
