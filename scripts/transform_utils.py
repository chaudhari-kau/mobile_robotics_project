#!/usr/bin/env python3

import numpy as np
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# coordinate transformation utilities for the controller
# these help with collision checking in various frames

def world_to_camera(curr_pose, curr_yaw, camera_position, world_point):
    """transform a point from world frame to camera frame
    
    this is equation 10 from the paper"""
    if curr_pose is None:
        return np.array([0.0, 0.0, 0.0])
    
    # rotation matrix from world to body
    c, s = np.cos(curr_yaw), np.sin(curr_yaw)
    r_wb = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    # translation vector
    t_wb = np.array([curr_pose.x, curr_pose.y, curr_pose.z])
    
    # transform point to body frame
    body_point = r_wb.T @ (np.array(world_point) - t_wb)
    
    # transform from body to camera frame (fixed transform)
    camera_point = body_point - camera_position
    
    return camera_point


def camera_to_world(curr_pose, curr_yaw, camera_position, camera_point):
    """transform a point from camera frame to world frame"""
    if curr_pose is None:
        return np.array([0.0, 0.0, 0.0])
    
    # transform from camera to body frame
    body_point = camera_point + camera_position
    
    # rotation matrix from body to world
    c, s = np.cos(curr_yaw), np.sin(curr_yaw)
    r_wb = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    # translation vector
    t_wb = np.array([curr_pose.x, curr_pose.y, curr_pose.z])
    
    # transform from body to world
    world_point = r_wb @ body_point + t_wb
    
    return world_point


def normalize_camera_coords(point, depth_max, hfov, vfov):
    """normalize camera coordinates according to equation 9 in paper"""
    x_norm = point[0] / depth_max
    y_norm = point[1] / (depth_max * np.tan(hfov))
    z_norm = point[2] / (depth_max * np.tan(vfov))
    
    return np.array([x_norm, y_norm, z_norm])


def unicycle_model(state, control, dt):
    """predict next state using unicycle dynamics model
    
    state: [x, y, z, yaw]
    control: [vx, wz, vz]"""
    x, y, z, yaw = state
    vx, wz, vz = control
    
    # update state - these r the kinematics from paper
    x_new = x + vx * np.cos(yaw) * dt
    y_new = y + vx * np.sin(yaw) * dt
    z_new = z + vz * dt
    yaw_new = yaw + wz * dt
    
    return np.array([x_new, y_new, z_new, yaw_new])
