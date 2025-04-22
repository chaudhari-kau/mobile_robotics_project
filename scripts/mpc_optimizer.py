#!/usr/bin/env python3

import numpy as np
from transform_utils import world_to_camera, unicycle_model, normalize_camera_coords
import torch
import rospy

class mpcoptimizer:
    """mpc-based trajectory optimization"""
    
    def __init__(self, prediction_horizon=10, dt=0.4, 
                 max_linear_vel=0.8, max_angular_vel=0.7,
                 collision_threshold=0.5, device='cuda'):
        self.N = prediction_horizon
        self.dt = dt
        self.max_linear_vel = max_linear_vel 
        self.max_angular_vel = max_angular_vel
        self.collision_threshold = collision_threshold
        self.device = device
        
        # weights for different cost components
        # tuned these by hand, might need tweaking
        self.w_collision = 1000.0   # collision penalty
        self.w_progress = 50.0      # goal progress reward
        self.w_control = 0.1        # control effort penalty
        
        # sampling parameters for optimization
        self.num_vx_samples = 5    # forward velocity samples
        self.num_wz_samples = 11   # angular velocity samples
        self.angle_margin = np.pi/3  # angular sampling margin
    
    def optimize(self, current_state, goal_position, latent_vector, collision_predictor, camera_params):
        # Extract parameters
        curr_pose = current_state[:3]
        curr_yaw = current_state[3]
        
        # Initialize target_angle with a default value
        target_angle = 0.0
        
        # Try to get angle to goal
        try:
            # Handle both Point objects and numpy arrays
            if hasattr(goal_position, 'x'):
                dx = goal_position.x - curr_pose[0]
                dy = goal_position.y - curr_pose[1]
            else:
                dx = goal_position[0] - curr_pose[0]
                dy = goal_position[1] - curr_pose[1]
                
            target_angle = np.arctan2(dy, dx)
        except Exception as e:
            rospy.logwarn(f"error calculating target angle: {e}, using default")
        
        # angle difference to goal
        angle_to_goal = target_angle - curr_yaw
        while angle_to_goal > np.pi:
            angle_to_goal -= 2*np.pi
        while angle_to_goal < -np.pi:
            angle_to_goal += 2*np.pi
        
        # sample control inputs - forward speed and turning rate
        vx_values = np.linspace(0.1, self.max_linear_vel, self.num_vx_samples)
        
        # angular velocities centered around goal direction
        wz_values = np.linspace(
            np.clip(angle_to_goal - self.angle_margin, -self.max_angular_vel, self.max_angular_vel),
            np.clip(angle_to_goal + self.angle_margin, -self.max_angular_vel, self.max_angular_vel),
            self.num_wz_samples
        )
        
        # tracking best solution
        best_cost = float('inf')
        best_trajectory = None
        best_control = None
        
        # evaluate each control sequence - this is the "multiple shooting" approach
        for vx in vx_values:
            for wz in wz_values:
                # maintain target height with p controller
                vz = 0.5 * (goal_position.z - curr_pose[2])
                vz = np.clip(vz, -0.5, 0.5)
                
                # control vector
                control = [vx, wz, vz]
                
                # simulate trajectory
                trajectory = [current_state.copy()]
                collision_scores = []
                
                # rollout over horizon (N steps)
                for step in range(self.N):
                    next_state = unicycle_model(
                        trajectory[-1], control, self.dt
                    )
                    trajectory.append(next_state)
                    
                    # check collision for this state - convert to camera frame
                    camera_point = world_to_camera(
                        curr_pose, curr_yaw, camera_position, next_state[:3]
                    )
                    
                    # skip points behind camera (conservative)
                    if camera_point[0] <= 0 or camera_point[0] > depth_max:
                        col_score = 1.0  # treat as collision
                    else:
                        # normalize point coordinates
                        point_norm = normalize_camera_coords(
                            camera_point, depth_max, hfov, vfov
                        )
                        
                        # predict collision using neural network
                        point_tensor = torch.tensor(
                            [point_norm], dtype=torch.float32
                        ).to(self.device)
                        
                        with torch.no_grad():
                            col_score = collision_predictor(
                                point_tensor, latent_vector
                            ).item()
                    
                    collision_scores.append(col_score)
                
                # compute total cost
                # 1. collision cost (any collision is bad)
                max_collision = max(collision_scores)
                collision_cost = self.w_collision if max_collision > self.collision_threshold else max_collision * 10.0
                
                # 2. goal progress
                end_state = trajectory[-1]
                end_dx = goal_position.x - end_state[0]
                end_dy = goal_position.y - end_state[1]
                end_distance = np.sqrt(end_dx*end_dx + end_dy*end_dy)
                
                start_dx = goal_position.x - current_state[0]
                start_dy = goal_position.y - current_state[1]
                start_distance = np.sqrt(start_dx*start_dx + start_dy*start_dy)
                
                progress = start_distance - end_distance
                progress_cost = self.w_progress * (1.0 - progress/start_distance)
                
                # 3. control effort
                control_cost = self.w_control * (vx*vx + 10.0*wz*wz)
                
                # total cost
                total_cost = collision_cost + progress_cost + control_cost
                
                # update best solution
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_trajectory = trajectory
                    best_control = control
        
        return best_control, best_trajectory
