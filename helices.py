import numpy as np
import utils
import scipy.optimize as opt
import scipy.stats as stats

class Helix:
    center = None
    radius = None
    phi = None
    omega = None
    Rot = None
    Rot_inv = None
    start_point = None

    def __init__(self):
        pass

    def solve(self, p1, p2, p3, B):
        """
        Solves a helix from three space points given the local direction of the B field
        ---
        p1, p2, p3  : array(3)  : space points in cartesian coordinates
        B           : array(3)  : direction of B field, magnitude unimportant
        """
        points = np.array([p1, p2, p3])
        dists = [np.sum(p**2) for p in points]
        self.start_point = points[np.argmax(dists)]

        def define_circle(p1, p2, p3):
            """
            Copy pasted from https://stackoverflow.com/questions/28910718/give-3-points-and-a-plot-circle
            """
            temp = p2[0] * p2[0] + p2[1] * p2[1]
            bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
            cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
            det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

            assert abs(det) > 1.0e-6, "Points in a straight line"

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

        Bnorm = B / np.sqrt(np.sum(B**2))
        self.Rot = rotation_matrix(Bnorm)
        self.Rot_inv = np.transpose(self.Rot)
        p1r = self.Rot @ p1
        p2r = self.Rot @ p2
        p3r = self.Rot @ p3
    
        self.center, self.radius = define_circle(p1r[:2], p2r[:2], p3r[:2])
        p1r -= self.center
        p2r -= self.center
        p3r -= self.center
    
        # Assume all points differ by no more than one complete arc
        rotated_points = np.array([p1r, p2r, p3r])
        rotated_points = rotated_points[np.argsort(rotated_points[:,-1])]

        angles = np.empty(rotated_points.shape[0])
        for i, p in enumerate(rotated_points):
            theta = np.arctan2(p[1], p[0])
            angles[i] = theta if theta >= 0 else 2*np.pi + theta
                                
        pos_angles = np.copy(angles)
        for i in [1, 2]:
            while pos_angles[i] < pos_angles[i-1]:
                pos_angles[i] += 2*np.pi
                                                                                
        neg_angles = np.copy(angles)
        for i in [1, 2]:
            while neg_angles[i] > neg_angles[i-1]:
                neg_angles[i] -= 2*np.pi
                                                                                                                        
        angles = pos_angles if abs(pos_angles[1] - pos_angles[0]) < abs(neg_angles[1] - neg_angles[0]) else neg_angles
                                                                                                                                        
        result = stats.linregress(rotated_points[:,-1], angles)
        self.omega, self.phi = result.slope, result.intercept
        
    def curve(self, t):
        """
        Parametric equation for the helix
        ---
        t       : float     : Parametric variable
        ---
        pos     : array(3)  : Spacial position of the helix in cartesian coordinates
        """
        x = self.radius * np.cos(self.omega * t + self.phi) + self.center[0]
        y = self.radius * np.sin(self.omega * t + self.phi) + self.center[1]
        z = t
        return self.Rot_inv @ np.array([x, y, z])

    def stepper(self, stepsize):
        """
        Generator function that yields spaces points on helix
        ---
        stepsize        : float       : Spacial stepsize between consecutive points on helix to be yielded
        ---
        point           : float(3)    : Next point on the helix
        """
        THRESHOLD_RADIUS = 1000 # Radius at which we assume the particle will make at most 1 precession.
        
        speed = np.sqrt(1 + (self.radius * self.omega)**2)
        delta_t = stepsize / speed

        start_rot = self.Rot @ self.start_point
        assert np.sum(self.start_point**2) == np.sum(start_rot**2)

        start_r = np.sqrt(np.sum(start_rot**2))
        if start_rot[2] < start_r and self.radius > THRESHOLD_RADIUS:
            start_phi_raw = np.arctan2(start_rot[1], start_rot[0])
            start_phi = start_phi_raw if start_phi_raw >= 0 else 2*np.pi + start_phi_raw
            t = (start_phi - self.phi) / self.omega
        else:
            t = start_rot[2]

        direction = np.int32(np.sum(self.curve(t)**2) < np.sum(self.curve(t + delta_t)**2))
        direction = 2 * direction - 1
    
        while True:
            yield self.curve(t), t
            t += direction * delta_t

def newton_intersection(helix, module, start_t):
    """
    Finds the intersection of a helix and a module using Newton's method
    ---
    helix   : Helix         : helix object
    module  : pd.Dataframe  : module specifications loaded from standard detector file
    start_t : float         : initial estimation parametric variable, i.e. helix.curve(start_t) is near the module
    ---
    pos     : array(3)      : position of the intersection
    """
    module_center = np.array([module.cx, module.cy, module.cz])
    module_rotation = np.array([
        [module.rot_xu, module.rot_xv, module.rot_xw],
        [module.rot_yu, module.rot_yv, module.rot_yw],
        [module.rot_zu, module.rot_zv, module.rot_zw]
    ])
    def curve_mf(t):
        pos_xyz = helix.curve(t)
        pos_uvw = np.transpose(module_rotation) @ (pos_xyz - module_center)
        return pos_uvw

    intersection_time = opt.newton(lambda t: curve_mf(t)[2], start_t) 
    
    # Check if intersection is within module boundaries
    u, v, w = curve_mf(intersection_time)
    if module.module_maxhu == module.module_minhu:
        within_module = abs(v) <= module.module_hv and abs(u) <= module.module_maxhu
    else:
        side_boundary = lambda x, side: side*(2*module.module_hv / (module.module_maxhu - module.module_minhu)) * (x - side*0.5*(module.module_maxhu + module.module_minhu))
        within_module = abs(v) <= module.module_hv and v >= side_boundary(u, 1) and v >= side_boundary(u, -1)

    if not within_module:
        return None
    return helix.curve(intersection_time)

def find_helix_intersection(helix, geometry, stepsize):
    """
    Find intersection between a helix and the next layer
    ---
    helix               : Helix         : helix object
    geometry            : pd.Dataframe  : dataframe with entries being all modules in the detector
    stepsize            : float         : step size with which to go along the helix before applying Newton's method, in mm.
    ---
    intersection        : array(3)      : spacial position of intersection
    intersection_vol    : int           : volume id for intersection layer
    intersection_lay    : int           : layer id for intersection layer
    """
    # Boundaries of the detector
    R_BOUND = 1050
    Z_BOUND = 3000

    stepper = helix.stepper(stepsize)
    init_pos, _ = next(stepper)
    init_vol, init_lay, _ = geometry.nearest_layer(init_pos)

    intersection = None
    intersection_vol = None
    intersection_lay = None
    while intersection is None:
        curr_pos, curr_time = next(stepper)
        if abs(curr_pos[2]) > Z_BOUND or np.sqrt(curr_pos[0]**2 + curr_pos[1]**2) > R_BOUND:
            break

        curr_vol, curr_lay, curr_dist = geometry.nearest_layer(curr_pos)
        if curr_vol is None:
            continue

        different_layer = curr_vol != init_vol or curr_lay != init_lay
        near_next_layer = different_layer and curr_dist < 2*stepsize
        
        if near_next_layer:
            near_modules = geometry.nearest_modules(curr_vol, curr_lay, curr_pos)
            for i in range(near_modules.shape[0]):
                module = near_modules.iloc[i]
                intersection = newton_intersection(helix, module, curr_time)
                if intersection is not None:
                    intersection_vol, intersection_lay, _ = geometry.nearest_layer(intersection)
                    assert intersection_vol == curr_vol and intersection_lay == curr_lay
                    break

    return intersection, intersection_vol, intersection_lay
