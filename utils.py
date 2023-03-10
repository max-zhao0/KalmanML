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
                    self.u_coord[(volume, layer)] = diameter / 2
                    
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
        --- 
        """
        return self.t_range.copy(), self.u_coord.copy()

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
        Get all hits near some point on a layer using a range of coordinates.
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
        end_phi = get_phi_coord(center[0] + area[0]) % lay_map.shape[0]
        start_t = max(get_t_coord(center[1] - area[1]), 0)
        end_t = min(get_t_coord(center[1] + area[1]), lay_map.shape[1] - 1)

        hits = []
        for t_coord in range(start_t, end_t + 1):
            phi_coord = start_phi
            while phi_coord != end_phi:
                hits += lay_map[phi_coord, t_coord]
                phi_coord = (phi_coord + 1) % lay_map.shape[0]

        return hits

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
        if volume in self.barrel_lst:
            s = self.u_coord[(volume, layer)]
            area[0] = radius / s
        else:
            area[0] = radius / center[1]
        area[1] = radius

        return self.get_near_hits(volume, layer, center, area)
 
def solve_helix(p1, p2, p3, B):
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
    print(omega12, omega23, omega13)
    
    omega = np.mean([omega12, omega23, omega13])
    phi = np.mean([phi12, phi23, phi13])
    
    return center, radius, omega, phi, Rot, Rot_inv

def helix_stepper(points, B, stepsize, start_index=None):
    """
    Generator function that yields spaces points on helix
    ---
    points      : float(3, 3) : Space points to fit helix to
    B           : float(3)    : Magnetic field vector
    stepsize    : float       : Spacial stepsize between consecutive points on helix to be yielded
    start_index : int         : Index of element in points to start yielding from. If none start from furthest from origin.
    ---
    point       : float(3)    : Next point on the helix
    """
    if start_index is None:
        dists = [np.sum(p**2) for p in points]
        start = points[np.argmax(dists)]
    else:
        start = points[start_index]
        
    center, radius, omega, phi, Rot, Rot_inv = solve_helix(points[0], points[1], points[2], B)
    speed = np.sqrt(1 + (radius * omega)**2)
    delta_t = stepsize / speed
    
    start_r = Rot @ start
    def helix(t):
        x = radius * np.cos(omega * t + phi) + center[0]
        y = radius * np.sin(omega * t + phi) + center[1]
        z = t
        return np.array([x, y, z])
    t = start_r[2]
    
    while True:
        yield Rot_inv @ helix(t)
        t += delta_t

if __name__ == "__main__":
     pass
     #detector_path = "/global/homes/m/max_zhao/mlkf/trackml/data/detectors.csv"
     #hits_path = "/global/homes/m/max_zhao/mlkf/trackml/data/hits.hdf5" 

     #loc = HitLocator(10, detector_path)
     #loc.load_hits(hits_path, 0)
     #print("Finished initializing")
     #hits = loc.get_near_hits(8, 8, (np.pi, 0), (np.pi / 7, 80))
     #t_range, u_coord = loc.get_detector_spec()
     #print(u_coord)
