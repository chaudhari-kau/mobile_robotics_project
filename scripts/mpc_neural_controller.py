#!/usr/bin/env python3

import torch
import numpy as np
import rospy
import os
import cv2
import time
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, Point, PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray

# import our modules
from neural_models import load_neural_models
from mpc_optimizer import mpcoptimizer
from transform_utils import world_to_camera, camera_to_world


class mpcneuralcontroller:
    def __init__(self):
        rospy.init_node('mpc_neural_controller')
        
        # camera parameters
        self.depth_max = 5.0
        self.hfov = 0.785  # ~45 degrees
        self.vfov = 0.442  # ~25 degrees
        self.camera_position = np.array([0.2, 0.0, -0.04])  # relative to drone body
        
        # mpc parameters - these match the paper
        self.prediction_horizon = 10
        self.dt = 0.4  # 4 sec horizon / 10 steps
        self.collision_threshold = 0.5
        
        # controller parameters
        self.target_height = 1.5
        self.max_linear_vel = 0.8
        self.max_angular_vel = 0.7
        
        # goal position - hardcoded for paper corridor
        self.goal_position = Point()
        self.goal_position.x = 9.0 
        self.goal_position.y = 0.0
        self.goal_position.z = 1.5
        
        # state variables
        self.state = "TAKEOFF"  # controller state machine
        self.curr_pose = None
        self.curr_orientation = None
        self.yaw = 0.0
        self.latent_vector = None
        self.depth_image = None
        self.received_depth = False
        self.debug_counter = 0
        
        # setup neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size_latent = 128  # latent space dimension
        self.model_path = os.path.expanduser("~/catkin_ws/src/colpred_nmpc/logs/inflated.pth")
        self.encoder, self.linear = load_neural_models(
            self.model_path, self.size_latent, self.device
        )
        
        # setup mpc optimizer
        self.mpc = mpcoptimizer(
            prediction_horizon=self.prediction_horizon,
            dt=self.dt,
            max_linear_vel=self.max_linear_vel,
            max_angular_vel=self.max_angular_vel,
            collision_threshold=self.collision_threshold,
            device=self.device
        )
        
        # ros interface
        self.bridge = CvBridge()
        self.depth_sub = rospy.Subscriber('/depth_image', Image, self.depth_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.odom_callback, queue_size=1)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.path_pub = rospy.Publisher('/predicted_path', Path, queue_size=1)
        self.marker_pub = rospy.Publisher('/collision_markers', MarkerArray, queue_size=1)
        self.rate = rospy.Rate(20)  # 20hz control loop
        
        # collision grid cache to avoid repeated predictions
        self.collision_grid = {}
        
        # oopsie here - forgot to check!
        # if self.encoder is None or self.linear is None:
        #    rospy.logerr("Failed to load neural networks!")
        #    return
        
        rospy.loginfo(f"mpc neural controller initialized on {self.device}")
    
    def depth_callback(self, msg):
        """process depth image and encode to latent space"""
        try:
            # convert ros message to numpy array
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            
            # resize if needed
            h, w = depth_img.shape
            target_h, target_w = 270, 480  # paper's dimensions
            if h != target_h or w != target_w:
                depth_img = cv2.resize(depth_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # normalize to [0,1]
            depth_img = np.clip(depth_img / self.depth_max, 0.0, 1.0)
            
            # store image
            self.depth_image = depth_img
            
            # convert to tensor
            depth_tensor = torch.from_numpy(depth_img.copy()).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # encode to latent space
            with torch.no_grad():
                self.latent_vector = self.encoder.encode(depth_tensor)
                
            self.received_depth = True
            
            # clear collision grid cache
            self.collision_grid = {}
            
        except Exception as e:
            rospy.logerr(f"error processing depth image: {e}")
    
    def odom_callback(self, msg):
        """process odometry data"""
        self.curr_pose = msg.pose.pose.position
        self.curr_orientation = msg.pose.pose.orientation
        
        # extract yaw from quaternion
        quat = [
            self.curr_orientation.x,
            self.curr_orientation.y,
            self.curr_orientation.z,
            self.curr_orientation.w
        ]
        euler = euler_from_quaternion(quat)
        self.yaw = euler[2]
        
        # log position occasionally
        if self.debug_counter % 20 == 0:
            rospy.loginfo(f"position: x={self.curr_pose.x:.2f}, y={self.curr_pose.y:.2f}, z={self.curr_pose.z:.2f}")
    
    def get_distance_to_goal(self):
        """calculate distance to goal position"""
        if self.curr_pose is None:
            return float('inf')
        
        dx = self.goal_position.x - self.curr_pose.x
        dy = self.goal_position.y - self.curr_pose.y
        dz = self.goal_position.z - self.curr_pose.z
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def publish_trajectory(self, trajectory):
        """publish predicted trajectory for visualization"""
        if not trajectory:
            return
            
        # create path message
        path = Path()
        path.header.frame_id = "world"
        path.header.stamp = rospy.Time.now()
        
        for state in trajectory:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = state[0]
            pose.pose.position.y = state[1]
            pose.pose.position.z = state[2]
            
            # convert yaw to quaternion
            quat = quaternion_from_euler(0, 0, state[3])
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            
            path.poses.append(pose)
        
        self.path_pub.publish(path)
    
    def run(self):
        """main control loop"""
        rospy.loginfo("starting mpc neural controller")
        
        # wait for first depth image
        timeout = 30.0  # seconds
        start_time = rospy.get_time()
        while not self.received_depth and not rospy.is_shutdown():
            if rospy.get_time() - start_time > timeout:
                rospy.logerr("timed out waiting for depth image")
                return
            rospy.loginfo_throttle(1, "waiting for depth image...")
            self.rate.sleep()
        
        rospy.loginfo("depth image received, starting control")
        
        while not rospy.is_shutdown():
            cmd = Twist()
            
            # check if reached goal
            distance_to_goal = self.get_distance_to_goal()
            if distance_to_goal < 0.8:
                rospy.loginfo("goal reached! stopping.")
                self.cmd_pub.publish(Twist())  # stop
                break
            
            # current state
            if self.curr_pose is None:
                self.rate.sleep()
                continue
                
            current_z = self.curr_pose.z
            
            # state machine for drone control
            if self.state == "TAKEOFF":
                # take off to target height
                cmd.linear.z = 0.5
                
                # transition when close to target height
                if current_z >= self.target_height - 0.2:
                    self.state = "MPC_CONTROL"
                    rospy.loginfo(f"takeoff complete at {current_z:.2f}m. starting mpc navigation.")
            
            elif self.state == "MPC_CONTROL":
                # prepare current state vector
                current_state = np.array([
                    self.curr_pose.x,
                    self.curr_pose.y,
                    self.curr_pose.z,
                    self.yaw
                ])
                
                # camera parameters for mpc
                camera_params = {
                    'depth_max': self.depth_max,
                    'hfov': self.hfov,
                    'vfov': self.vfov,
                    'position': self.camera_position
                }
                
                # solve mpc problem
                best_control, best_trajectory = self.mpc.optimize(
                    current_state, self.goal_position, 
                    self.latent_vector, self.linear, 
                    camera_params
                )
                
                if best_control is not None:
                    # apply first control action
                    cmd.linear.x = best_control[0]
                    cmd.angular.z = best_control[1]
                    cmd.linear.z = best_control[2]
                    
                    # publish trajectory visualization
                    self.publish_trajectory(best_trajectory)
                    
                    # log commands
                    rospy.loginfo_throttle(1, f"mpc: vx={cmd.linear.x:.2f}, wz={cmd.angular.z:.2f}, vz={cmd.linear.z:.2f}")
                else:
                    # fallback if mpc fails
                    rospy.logwarn("mpc optimization failed, using safe fallback control")
                    cmd.linear.x = 0.1
                    cmd.angular.z = 0.5
                    cmd.linear.z = 0.5 * (self.target_height - current_z)
            
            # safety limits
            cmd.linear.x = np.clip(cmd.linear.x, 0.0, self.max_linear_vel)
            cmd.linear.z = np.clip(cmd.linear.z, -0.5, 0.5)
            cmd.angular.z = np.clip(cmd.angular.z, -self.max_angular_vel, self.max_angular_vel)
            
            # publish command
            self.cmd_pub.publish(cmd)
            self.debug_counter += 1
            
            self.rate.sleep()


if __name__ == '__main__':
    try:
        controller = mpcneuralcontroller()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"error in controller: {e}")
        # emergency stop
        try:
            pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            pub.publish(Twist())
        except:
            pass
