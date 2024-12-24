cmake_minimum_required(VERSION 3.5)
project(aruco_detect)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fiducial_msgs REQUIRED)

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Python setup (equivalent in ROS2)
install(PROGRAMS scripts/create_markers.py
  DESTINATION lib/${PROJECT_NAME}
)

# Build the executable
add_executable(aruco_detect src/aruco_detect.cpp)
ament_target_dependencies(aruco_detect 
  rclcpp 
  tf2 
  tf2_ros 
  tf2_geometry_msgs 
  visualization_msgs 
  image_transport 
  cv_bridge 
  sensor_msgs 
  std_msgs 
  vision_msgs 
  std_srvs
  fiducial_msgs 
  OpenCV
)

# Install executable
install(TARGETS aruco_detect
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# # Add testing (adjusted for ROS2)
# if(BUILD_TESTING)
#   find_package(ament_cmake_gtest REQUIRED)

#   ament_add_gtest(aruco_images_test
#     test/aruco_images_test.cpp
#   )
#   target_link_libraries(aruco_images_test ${OpenCV_LIBS})
# endif()

# Package exports
ament_package()
