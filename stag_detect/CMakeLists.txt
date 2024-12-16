cmake_minimum_required(VERSION 3.5)
project(stag_detect)

# Required dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(fiducial_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dynamic_reconfigure QUIET) # Not fully supported in ROS2, alternatives may be needed
find_package(Threads REQUIRED)

# Generate dynamic reconfigure options (if supported in ROS2)
# NOTE: ROS2 doesn't have built-in support for dynamic_reconfigure like ROS1.
# You will need to use an alternative or manually adapt the configuration system.

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Declare the library
add_library(stag_core
  src/stag/Decoder.cpp
  src/stag/Drawer.cpp
  src/stag/EDInterface.cpp
  src/stag/Ellipse.cpp
  src/stag/Marker.cpp
  src/stag/PoseRefiner.cpp
  src/stag/Quad.cpp
  src/stag/QuadDetector.cpp
  src/stag/Stag.cpp
  src/stag/utility.cpp
  src/stag/ED/ED.cpp
  src/stag/ED/EDInternals.cpp
  src/stag/ED/EDLines.cpp
  src/stag/ED/GradientOperators.cpp
  src/stag/ED/ImageSmooth.cpp
  src/stag/ED/LineSegment.cpp
  src/stag/ED/MyMath.cpp
  src/stag/ED/NFA.cpp
  # src/stag/ED/Utilities.cpp
  src/stag/ED/ValidateEdgeSegments.cpp
  include/stag_ros/load_yaml_tags.h
)

ament_target_dependencies(stag_core
  OpenCV
)

# Declare the executable
add_executable(stag_detect src/stag_ros/stag_detect.cpp)
ament_target_dependencies(stag_detect
  camera_info_manager
  cv_bridge
  geometry_msgs
  tf2_geometry_msgs
  tf2_ros
  tf2
  visualization_msgs
  vision_msgs
  image_transport
  sensor_msgs
  std_msgs
  fiducial_msgs
  OpenCV
)

target_link_libraries(stag_detect
  stag_core
  ${OpenCV_LIBS}
)

# Install targets
install(TARGETS
  stag_core
  stag_detect
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

# Install directories
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY cfg/
  DESTINATION share/${PROJECT_NAME}/cfg
)

# Export package
ament_package()
