import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch # Switch occurs at t_final - t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        # Hint: Both self.traj_controller and self.pose_controller have compute_control() functions. 
        #       When should each be called? Make use of self.t_before_switch and 
        #       self.traj_controller.traj_times.
        ########## Code starts here ##########
        if t < self.t_before_switch:
            # use trajectory controller
            return self.traj_controller.compute_control(x, y, th, t)
        else:
            # use pose regulation
            return self.pose_controller.compute_control(x, y, th, t)
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (list of [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # Hint 1 - Determine nominal time for each point in the path using V_des
    # Hint 2 - Use splrep to determine cubic coefficients that best fit given path in x, y

    path = np.array(path)
    x_old = path[:,0]
    y_old = path[:,1]
    n = np.shape(x_old)[0]
    t_segment = np.zeros((n,))
    t_segment[1:] = np.linalg.norm(np.column_stack((x_old[1:] - x_old[:-1], y_old[1:] - y_old[:-1])),
                                   axis = 1) / V_des
    t_path = np.cumsum(t_segment, dtype=float)
    x_trajectory = scipy.interpolate.splrep(t_path, x_old, s = alpha)
    y_trajectory = scipy.interpolate.splrep(t_path, y_old, s = alpha)

    t_smoothed = np.arange(0.0, max(t_path), dt)
    x_d = scipy.interpolate.splev(t_smoothed, x_trajectory)
    y_d = scipy.interpolate.splev(t_smoothed, y_trajectory)
    xd_d = scipy.interpolate.splev(t_smoothed, x_trajectory, der = 1)
    xdd_d = scipy.interpolate.splev(t_smoothed, x_trajectory, der = 2)
    yd_d = scipy.interpolate.splev(t_smoothed, y_trajectory, der = 1)
    ydd_d = scipy.interpolate.splev(t_smoothed, y_trajectory, der = 2)
    theta_d = np.arctan2(yd_d, xd_d)
    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    Hint: Take a close look at the code within compute_traj_with_limits() and interpolate_traj() 
          from P1_differential_flatness.py
    """
    ########## Code starts here ##########
    V, om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    x_f, y_f, _, xd_f, yd_f, _, _ = traj[-1]
    s_f = State(x_f, y_f,
                np.sqrt(xd_f ** 2 + yd_f ** 2),
                np.arctan2(yd_f, xd_f))
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
