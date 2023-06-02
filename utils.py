import numpy as np
import h5py

class HitLocator:
    # Volume ids in the detector
    volume_lst = np.array([7, 8, 9, 12, 13, 14, 16, 17, 18])
    # Volumes that are barrel shaped
    barrel_set = {8, 13, 17}

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
        
        for volume in self.volume_lst:
            self.layer_dict[volume] = []
            
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

                t = z if volume in self.barrel_set else np.sqrt(x**2 + y**2)
                
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
        if volume in self.barrel_set:
            s = self.u_coord[(volume, layer)]
            area[0] = radius / s
        else:
            area[0] = radius / center[1]
        area[1] = radius

        return self.get_near_hits(volume, layer, center, area)

def DEPRECATED_solve_helix(p1, p2, p3, B):
    """
    Gives the parameters of a circular helix fitted to 3 points.
    ---
    p1, p2, p3 : float(3)    : Space points to be fit to
    B          : float(3)    : Magnetic field vector
    ---
    center     : float(3)    : center of helical cylinder in rotated coordinates
    radius     : float       : radius of helical cylinder in rotated coordinates
    omega      : float       : angular velocity of helix
    phi        : float       : phase of helix
    Rot        : float(3, 3) : Rotation matrix such that B points in z direction
    Rot_inv    : float(3, 3) : Inverse of Rot
    """
    def define_circle(p1, p2, p3):
        """
        Copy pasted from https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return (None, np.inf)

        # Center of circle
        cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        return np.array([cx, cy, 0]), radius
    
    def rotation_matrix(a, b=np.array([0, 0, 1])):
        if (a == b).all():
            return np.identity(3)
        v = np.cross(a, b)
        s = np.sqrt(np.sum(v**2))
        c = np.dot(a, b)
        
        I = np.identity(3)
        v_x = np.array([
            [0    , -v[2], v[1] ],
            [v[2] , 0    , -v[0]],
            [-v[1], v[0] , 0    ]
        ])
        
        return I + v_x + ((1 - c)/s**2) * (v_x @ v_x)
    
    def fit_helix(p0, p1):
        """
        Based on technique from https://www.geometrictools.com/Documentation/HelixFitting.pdf
        """
        #delta = p1[0]**2 * p2[1]**2 - p2[0]**2 * p1[1]**2
        theta_0 = np.arctan(p0[1] / p0[0])
        theta_1 = np.arctan(p1[1] / p0[0])
        omega = (theta_1 - theta_0) / (p1[2] - p0[2])
        phi = (theta_0 * p1[2] - theta_1 * p0[2]) / (p1[2] - p0[2])
        return omega, phi
        
    Bnorm = B / np.sqrt(np.sum(B**2))
    Rot = rotation_matrix(Bnorm)
    Rot_inv = np.transpose(Rot)
    p1r = Rot @ p1
    p2r = Rot @ p2
    p3r = Rot @ p3
    
    center, radius = define_circle(p1r[:2], p2r[:2], p3r[:2])
    p1r -= center
    p2r -= center
    p3r -= center
    omega12, phi12 = fit_helix(p1r, p2r)
    omega23, phi23 = fit_helix(p2r, p3r)
    omega13, phi13 = fit_helix(p1r, p3r)
    
    omega = np.mean([omega12, omega23, omega13])
    phi = np.mean([phi12, phi23, phi13])
    
    return center, radius, omega, phi, Rot, Rot_inv

def helix_stepper(points, B, stepsize, save_helices=None, start_index=None):
    """
    Generator function that yields spaces points on helix
    ---
    points          : float(3, 3) : Space points to fit helix to
    B               : float(3)    : Magnetic field vector
    stepsize        : float       : Spacial stepsize between consecutive points on helix to be yielded
    save_helices    : list        : Save helix solutions
    start_index     : int         : Index of element in points to start yielding from. If none start from furthest from origin.
    ---
    point       : float(3)    : Next point on the helix
    """
    THRESHOLD_RADIUS = 1000 # Radius at which we assume the particle will make at most 1 precession.

    if start_index is None:
        dists = [np.sum(p**2) for p in points]
        start = points[np.argmax(dists)]
    else:
        start = points[start_index]
        
    center, radius, omega, phi, Rot, Rot_inv = solve_helix(points[0], points[1], points[2], B)
    speed = np.sqrt(1 + (radius * omega)**2)
    delta_t = stepsize / speed

    start_rot = Rot @ start
    assert np.sum(start**2) == np.sum(start_rot**2)

    def helix(t):
        x = radius * np.cos(omega * t + phi) + center[0]
        y = radius * np.sin(omega * t + phi) + center[1]
        z = t
        return Rot_inv @ np.array([x, y, z])
    
    start_r = np.sqrt(np.sum(start_rot**2))
    if start_rot[2] < start_r and radius > THRESHOLD_RADIUS:
        start_phi_raw = np.arctan2(start_rot[1], start_rot[0])
        start_phi = start_phi_raw if start_phi_raw >= 0 else 2*np.pi + start_phi_raw
        t = (start_phi - phi) / omega
    else:
        t = start_rot[2]

    direction = np.int32(np.sum(helix(t)**2) < np.sum(helix(t + delta_t)**2))
    direction = 2 * direction - 1
    
    if save_helices is not None:
        solution = np.empty((3+3+3+3+1+1+1))
        solution[0:9] = points.flatten()
        solution[9:12] = center
        solution[12] = radius
        solution[13] = omega
        solution[14] = phi
        save_helices.append(solution)

    while True:
        yield helix(t)
        t += direction * delta_t

class BFieldMap:         
    def get(self, pos):
        return np.array([0, 0, 2])
