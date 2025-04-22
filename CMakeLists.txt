cmake_minimum_required(VERSION 3.0.2)
project(mobile_robotics_project)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  gazebo_ros
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch custom_worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
