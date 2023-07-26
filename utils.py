import numpy as np
import h5py
import pandas as pd

class Geometry:
    # Volume ids in the detector
    VOLUMES = np.array([7, 8, 9, 12, 13, 14, 16, 17, 18])
    # Volumes that are barrel shaped
    BARRELS = {8, 13, 17}
    Z_BUFFER = 50
    R_BUFFER = 25

    detector = None
    bmap = None
    volume_bounds = {}
    layer_bounds = {}
    detector_map = {}

    def __init__(self, detector_path, bmap):
        self.detector = pd.read_csv(detector_path)
        self.bmap = bmap

        for vol in self.VOLUMES:
            vol_modules = self.detector.loc[self.detector.volume_id == vol]
            layer_ids = set(vol_modules.layer_id)
            self.detector_map[vol] = {}

            r_min = np.inf
            r_max = -np.inf
            z_min = np.inf
            z_max = -np.inf
            
            for lay in layer_ids:
                lay_modules = vol_modules.loc[vol_modules.layer_id == lay]
                self.detector_map[vol][lay] = lay_modules

                lay_min = np.inf
                lay_max = -np.inf

                for i in range(lay_modules.shape[0]):
                    module = lay_modules.iloc[i]
                    rotation_matrix = np.array([
                        [module.rot_xu, module.rot_xv, module.rot_xw],
                        [module.rot_yu, module.rot_yv, module.rot_yw],
                        [module.rot_zu, module.rot_zv, module.rot_zw]
                    ])
                    center = np.array([module.cx, module.cy, module.cz])
                    
                    transform_xyz = lambda uvw: rotation_matrix @ uvw + center
                    for pm_du, dv in [(module.module_maxhu, module.module_hv), (module.module_minhu, -module.module_hv)]:
                        for du in [pm_du, -pm_du]:
                            corner = transform_xyz(np.array([du, dv, 0]))
                            corner_r = np.sqrt(np.sum(corner[:2]**2))
                            corner_z = corner[2]
                            if corner_r > r_max:
                                r_max = corner_r
                            elif corner_r < r_min:
                                r_min = corner_r
                            if corner_z > z_max:
                                z_max = corner_z
                            elif corner_z < z_min:
                                z_min = corner_z 

                            corner_lay = corner_r if vol in self.BARRELS else corner_z
                            if corner_lay > lay_max:
                                lay_max = corner_lay
                            elif corner_lay < lay_min:
                                lay_min = corner_lay
                
                if vol not in self.layer_bounds:
                    self.layer_bounds[vol] = {}
                self.layer_bounds[vol][lay] = np.array([lay_min, lay_max])

            r_min -= self.R_BUFFER
            r_max += self.R_BUFFER
            z_min -= self.Z_BUFFER
            z_max += self.Z_BUFFER
            self.volume_bounds[vol] = np.array([r_min, r_max, z_min, z_max])

    def nearest_layer(self, point):
        assert len(point) == 3
        point_r = np.sqrt(point[0]**2 + point[1]**2)
        point_z = point[2]

        volume = None
        for vol in self.volume_bounds:
            if self.volume_bounds[vol][0] <= point_r <= self.volume_bounds[vol][1] and self.volume_bounds[vol][2] <= point_z <= self.volume_bounds[vol][3]:
                volume = vol
                break
        
        if volume is None:
            layer = None
            distance = None
        else:
            distance = np.inf
            point_s = point_r if vol in self.BARRELS else point_z
            
            for lay in self.layer_bounds[vol]:
                lay_dist = abs(np.mean(self.layer_bounds[vol][lay]) - point_s)
                if lay_dist < distance:
                    distance = lay_dist
                    layer = lay

        return volume, layer, distance

    def nearest_modules(self, volume, layer, point, nmodules=4):
        lay_modules = self.detector_map[volume][layer]
        square_distances = (lay_modules.cx - point[0])**2 + (lay_modules.cy - point[1])**2 + (lay_modules.cz - point[2])**2
        indices = np.argpartition(square_distances, nmodules)
        closest_modules = lay_modules.iloc[indices[:nmodules]]
        return closest_modules

class HitLocator: 
    hit_map = {}
    # Range of z for barrels, r for endcaps. Defined on volumes.
    t_range = {}
    # r for barrels, z for endcaps. Defined on layers.
    u_coord = {}
    # list of layers for each volume
    layer_dict = {}
    # arrays of all hits on layer
    full_layers = {}

    def __init__(self, resolution, detector_path):
        """
        Initialize the data structure with empty cells using detector geometry file.
        Detector geometry file can be found at https://www.kaggle.com/competitions/trackml-particle-identification/data.
        ---
        resolution      : float     : width of cells
        detector_path   : String    : path to detector csv file.
        """
        volume_id, layer_id, module_id, cx, cy, cz, hv = np.loadtxt(detector_path, delimiter=",", skiprows=1, usecols=[0,1,2,3,4,5,18], unpack=True)
        
        for volume in Geometry.VOLUMES:
            self.layer_dict[volume] = []
            
            lay_id_vol = layer_id[volume_id == volume]
            cx_vol = cx[volume_id == volume]
            cy_vol = cy[volume_id == volume]
            cz_vol = cz[volume_id == volume]
            hv_vol = hv[volume_id == volume]
            max_hv = max(hv_vol)

            vol_map = {}
            if volume in Geometry.BARRELS:
                min_z = min(cz_vol) - max_hv
                max_z = max(cz_vol) + max_hv
                self.t_range[volume] = (min_z, max_z)
                
                for layer in set(lay_id_vol):
                    self.layer_dict[volume].append(layer)
                    
                    cx_lay = cx_vol[lay_id_vol == layer]
                    cy_lay = cy_vol[lay_id_vol == layer]
                    diameter = 2 * np.sqrt(cx_lay[0]**2 + cy_lay[0]**2)
                    self.u_coord[(volume, layer)] = diameter / 2
                    
                    z_dim = round(np.ceil((max_z - min_z) / resolution))
                    phi_dim = round(np.ceil(np.pi * diameter / resolution))
                    vol_map[layer] = np.empty((phi_dim, z_dim), dtype=list)
            else:
                cr_vol = np.sqrt(cx_vol**2 + cy_vol**2)
                min_r = min(cr_vol) - max_hv
                max_r = max(cr_vol) + max_hv
                
                if volume == 16 or volume == 18:
                    # Unfortunate hack
                    max_r += 6
                elif volume == 7 or volume == 9:
                    max_r += 1
                elif volume == 12 or volume == 14:
                    max_r += 1

                self.t_range[volume] = (min_r, max_r)

                r_dim = round(np.ceil((max_r - min_r) / resolution))
                phi_dim = round(np.ceil(np.pi * (max_r + min_r) / resolution))
                
                for layer in set(lay_id_vol):
                    self.layer_dict[volume].append(layer)
                    
                    cz_lay = cz_vol[lay_id_vol == layer]
                    assert abs(min(cz_lay) - max(cz_lay)) < 12
                    self.u_coord[(volume, layer)] = np.mean(cz_lay)

                    vol_map[layer] = np.empty((phi_dim, r_dim), dtype=list)

            for layer in set(lay_id_vol):
                for row in vol_map[layer]:
                    for i in range(len(row)):
                        row[i] = []

            self.hit_map[volume] = vol_map

    def get_detector_spec(self):
        """
        Gets detector info
        ---
        t_range     : dict      : indexed by volume, gives range of z/r depending on if barrel or endcap respectively
        u_coord     : dict      : indexed by (volume, layer), gives coordinate that specifies layers position: r/z respectively for barrel and endcap.
        layer_dict  : dict      : indexed by volume, gives list of layers in each volume.
        """
        return self.t_range.copy(), self.u_coord.copy(), self.layer_dict.copy()

    def load_hits(self, hits_path, event_id, layers_to_save={(8,2), (8,4), (7,14), (9,2), (8,6), (7,12), (9,4)}):
        """
        Load hits into data structure from hits file
        ---
        hits_path       : String        : path to hits file
        event_id        : int           : event to store
        layers_to_save  : set (tuples)  : layers that should be stored in full arrays
        """
        volumes_to_save = {vol for vol, lay in layers_to_save}

        event_id = str(event_id)
        f = h5py.File(hits_path, "r")

        for volume_id in f[event_id].keys():
            volume = int(volume_id)
            vol_range = self.t_range[volume]
            vol_hits = f[event_id + "/" + volume_id + "/hits"]

            if volume in volumes_to_save:
                for layer in self.layer_dict[volume]:
                    if (volume, layer) in layers_to_save:
                        self.full_layers[(volume, layer)] = vol_hits[vol_hits[:,0] == layer]

            for hit in vol_hits:
                layer = hit[0]
                lay_map = self.hit_map[volume][layer]
                x, y, z = hit[6:9]

                raw_phi = np.arctan2(y, x)
                phi = raw_phi if raw_phi >= 0 else 2 * np.pi + raw_phi
                phi_coord = round((phi / (2 * np.pi)) * (lay_map.shape[0] - 1))

                t = z if volume in Geometry.BARRELS else np.sqrt(x**2 + y**2)
                
                assert vol_range[0] <= t <= vol_range[1], str(t) + " Vol: " + str(volume) + " " + str(vol_range)
                t_coord = round((lay_map.shape[1] - 1) * (t - vol_range[0]) / (vol_range[1] - vol_range[0]))

                lay_map[phi_coord, t_coord].append(hit)

        f.close()

    def get_layer_hits(self, volume, layer):
        """
        Get all hits on a layer
        ---
        volume  : int
        layer   : int
        """
        return self.full_layers[volume, layer]

    def get_near_hits(self, volume, layer, center, area):
        """
        Get all hits near some point on a layer using a range of coordinates.
        ---
        volume  : int               : volume that contains the layer
        layer   : int               : layer number
        center  : (float, float)    : point around which to collect hits. Of the form (phi, t) where t = z if barrel volume and r if endcap
        area    : (float, float)    : range with which to collect hits. Essentially collect hits with coordinate in center +- area
        ---
        Returns:
        hits    : array             : array of hits
        """
        assert area[0] > 0 and area[1] > 0

        lay_map = self.hit_map[volume][layer]
        lay_range = self.t_range[volume]

        get_phi_coord = lambda phi: round((lay_map.shape[0] - 1) * phi / (2 * np.pi))
        get_t_coord = lambda t: round((lay_map.shape[1] - 1) * (t - lay_range[0]) / (lay_range[1] - lay_range[0]))

        start_phi = get_phi_coord(center[0] - area[0]) % lay_map.shape[0]
        end_phi = get_phi_coord(center[0] + area[0]) % lay_map.shape[0]
        start_t = max(get_t_coord(center[1] - area[1]), 0)
        end_t = min(get_t_coord(center[1] + area[1]), lay_map.shape[1] - 1)

        hits = []
        for t_coord in range(start_t, end_t + 1):
            phi_coord = start_phi
            while phi_coord != end_phi:
                hits += lay_map[phi_coord, t_coord]
                phi_coord = (phi_coord + 1) % lay_map.shape[0]

        return np.array(hits)

    def get_hits_around(self, volume, layer, center, radius):
        """
        Gets all hits around a point defined by a charicteristic distance.
        ```
        volume  : int               : volume that contains the layer
        layer   : int               : layer number
        center  : (float, float)    : point around which to collect hits. Of the form (phi, t) where t = z if barrel volume and r if endcap
        radius  : float             : radius around which to collect hits.
        ```
        Returns:
        hits    : List              : list of hits
        """
        area = np.empty(2)
        if volume in Geometry.BARRELS:
            s = self.u_coord[(volume, layer)]
            area[0] = radius / s
        else:
            area[0] = radius / center[1]
        area[1] = radius

        return self.get_near_hits(volume, layer, center, area)

class BFieldMap:         
    def get(self, pos):
        return np.array([0, 0, 2])
