#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImprovedImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        # Subscribe to the camera topic from Hector quadrotor
        self.image_sub = rospy.Subscriber('/front_cam/camera/image', Image, self.image_callback, queue_size=1)
        # Publisher for visualization
        self.processed_pub = rospy.Publisher('/processed_image', Image, queue_size=1)
        # Publisher for the collision avoidance system
        self.depth_pub = rospy.Publisher('/depth_image', Image, queue_size=1)
        
        self.debug_counter = 0
        self.last_image = None
        
        rospy.init_node('image_processor', anonymous=True)
        rospy.loginfo("Improved image processor initialized")
        
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Convert RGB to grayscale
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better contrast
            gray_image = cv2.equalizeHist(gray_image)
            
            # Normalize to 0-1 range
            normalized = gray_image.astype(np.float32) / 255.0
            
            # Simulate depth based on image brightness
            # Invert so brighter areas (obstacles) have higher values
            simulated_depth = 1.0 - normalized
            
            # Apply Gaussian blur to smooth the depth image
            simulated_depth = cv2.GaussianBlur(simulated_depth, (5, 5), 0)
            
            # Edge enhancement to better detect obstacle boundaries
            if self.last_image is not None:
                edges = cv2.Canny(gray_image, 50, 150)
                edges_f = edges.astype(np.float32) / 255.0
                # Emphasize edges in the depth map
                simulated_depth = np.maximum(simulated_depth, edges_f * 0.8)
            
            # Apply a depth window to reduce noise
            simulated_depth = np.clip(simulated_depth, 0.3, 0.95)
            
            # Store for next frame processing
            self.last_image = gray_image
            
            # For visualization - convert to 8-bit
            display_image = (normalized * 255).astype(np.uint8)
            
            # Log image statistics occasionally
            self.debug_counter += 1
            if self.debug_counter % 100 == 0:
                min_val = np.min(simulated_depth)
                max_val = np.max(simulated_depth)
                mean_val = np.mean(simulated_depth)
                rospy.loginfo(f"Depth image stats - Min: {min_val:.3f}, Max: {max_val:.3f}, Mean: {mean_val:.3f}")
            
            # Publish processed grayscale image for visualization
            processed_msg = self.bridge.cv2_to_imgmsg(display_image, encoding="mono8")
            processed_msg.header = msg.header
            self.processed_pub.publish(processed_msg)
            
            # Publish simulated depth image for collision avoidance
            depth_msg = self.bridge.cv2_to_imgmsg(simulated_depth, encoding="32FC1")
            depth_msg.header = msg.header
            self.depth_pub.publish(depth_msg)
            
            rospy.loginfo_throttle(1, "Processed image")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            
def main():
    processor = ImprovedImageProcessor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
