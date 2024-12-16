cmake_minimum_required(VERSION 3.5)

project(fiducial_msgs)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Message and service generation
find_package(rosidl_default_generators REQUIRED)

# Define message files
set(msg_files
  "msg/Fiducial.msg"
  "msg/FiducialArray.msg"
  "msg/FiducialTransform.msg"
  "msg/FiducialTransformArray.msg"
  "msg/FiducialMapEntry.msg"
  "msg/FiducialMapEntryArray.msg"
)

# Define service files
set(srv_files
  "srv/InitializeMap.srv"
)

# Generate messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs
)

# Package exports
ament_export_dependencies(rosidl_default_runtime)

# Install
install(
  DIRECTORY msg srv
  DESTINATION share/${PROJECT_NAME}
)

# Ament package declaration
ament_package()
