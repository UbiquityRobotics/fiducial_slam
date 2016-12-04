
# Simultaneous Localization and Mapping Using Fiducial Markers
Travis:  [![Build Status](https://travis-ci.org/UbiquityRobotics/fiducials.svg?branch=kinetic-devel)](https://travis-ci.org/UbiquityRobotics/fiducials)

Jenkins: [![Build Status](http://build.ros.org/view/Kdev/job/Kdev__fiducials__ubuntu_xenial_amd64/badge/icon)](http://build.ros.org/view/Kdev/job/Kdev__fiducials__ubuntu_xenial_amd64/)

## Overview

This package implements a system that uses ceiling mounted
fiducial markers (think QR Codes) to allow a robot to identify
its location and orientation.  It does this by constructing
a map of the ceiling fiducials.  The position of one fiducial
needs to be specified, then a map of the fiducials is built 
up from observing pairs of markers in the same image. 
Once the map has been constructed, the robot can identify
its location by locating itself relative to one or more 
ceiling fiducials.


## Installing software

First you must install ROS (Robot Operating System),
see [install ROS](http://wiki.ros.org/ROS/Installation) for more details.

To install from binary packages:

     $ sudo apt-get install ros-install-fiducials

Or, alternatively, to build from source:

Create a ROS catkin workspace, if you don't already have one:

     $ mkdir -p ~/catkin_ws/src
     $ cd ~/catkin_ws/src
     $ catkin_init_workspace

Then get the source:

    $ git clone https://github.com/UbiquityRobotics/fiducials 
    $ git clone https://github.com/UbiquityRobotics/raspicam_node 
    $ cd ..
    $ catkin_make    


Install inkscape

     $ sudo apt-get install inkscape

## Generating fiducials

The fiducials are typically generated from the laptop/desktop.

Fiducial tags are generated using the following commands:

	$ cd /tmp
        $ rosrun fiducial_lib Tags num ...

Thus, the following command will generate `tag42.svg` and `tag43.svg`:

        $ rosrun fiducial_lib Tags 42 43

To generate a 100 at a time, use the following command:

        $ rosrun fiducial_lib Tags 17{0,1,2,3,4,5,6,7,8,9}{0,1,2,3,4,5,6,7,8,9}

This will generate `tag1700.svg` through `tag1799.svg`.

To convert the `.svg` files to to `.pdf` files use the`inkscape`
program and the following commands:

Convert a single .svg to a single .pdf:

        $ inkscape --without-gui --export-pdf=tag42.pdf tag42.svg

Convert a bunch at a time:

        $ for n in 17{0,1,2,3,4,5,6,7,8,9}{0,1,2,3,4,5,6,7,8,9} ; do \
           inkscape --without-gui --export-pdf=tag$n.pdf tag$n.svg ; \
           done

Merge the pdf files:

        $ sudo apt-get install poppler-utils
        $ pdfunite tag17??.pdf tags17xx.pdf
        $ rm tag*.svg

You can print the .pdfs using using `evince`:

	$ sudo apt-get install evince
        $ evince tags17xx.pdf

## Map Creation

To create an empty map file with fiducial 301 (the id of 301 is arbitrary) at the origin:

        $ mkdir -p ~/.ros/slam
        $ echo '301 0.0 0.0 0.0 180.0 0.0 180.0 0.0 1' > ~/.ros/slam/map.txt

This assumes the fiducial is on the ceiling, and the 180 degree rotations are used
so that the pose of fiducials is determined in the co-ordinate system of the floor.  
Increased accuracy can be gained by specifying the height of the fiducial (z), 
and that this can also be determined from the output of `fiducial_slam.py`.  

It is important that the transform from base link to the camera frame is
correctly specified.  In the `magni_robot` package, this is specified in the URDF.

## Running Fiducials

1. First create an empty map as above.

2. In order to see the fiducials during map building, run rviz:

        $ roslaunch fiducial_slam fiducial_rviz.launch

Note that the robot node should be running (so that the transform from odom to base_link is published)
and at at least one of the fiducials that is in the the map should have been observed before rviz 
can display anything in the map frame.

        $ roslaunch fiducial_detect fiducial_detect.launch
        $ roslaunch fiducial_slam fiducial_slam.launch
		
At this point the robot should be capable of executing move_base and other navigation functions.		

## Visualization with rviz

While building the map, you should start to see something that looks as follows:

> ![rviz Showing Fiducials](fiducials_rviz.png "rviz Displaying Fiducials")

* Red cubes represent fiducials that are currently in view
  of the camera.

* Green cubes represent fiducials that are in the map, but
  not currently in the view of the camera.

* Blue lines connect pairs of fiducials that have shown
  up in the camera view at the same time.  The map is constructed
  by combining fiducial pairs.
  

### Tags

The Tags program is used to generate .svg files for tags.  Running Tags:

    Tags 41 42

will generate tag41.svg and tag42.svg.  To print:

    inkscape --without-gui --export-pdf=tag41.pdf tag41.svg

	
## Nodes

### fiducial_detect fiducial_detect

This node finds fiducial markers in images stream and publishes their vertices
(corner points) and estimates 3D transforms from the camera to the fiducials.
It also has 2D SLAM built in.

#### Parameters

**estimate_pose** If `true`, 3D pose estimation is performed and fiducial
transforms are published. Default `true`.

**fiducial_len** The length of one side of a fiducial in meters, used by the
pose estimation.  Default 0.146.

**undistort_points** If `false`, it is assumed that the input is an undistorted
image, and the vertices are used directly to calculate the fiducial transform.
If it is `true`, then the vertices are undistorted first. This is faster, but
less accurate.  Default `false`.


#### Published Topics


**/fiducial_vertices** A topic of `fiducial_detect/Fiducial*` messages with the detected
fiducial vertices.


**/fiducial_transforms** A topic of `fiducial_pose/FiducialTransform` messages 
with the computed fiducial pose.

#### Subscribed Topics

**camera** An `ImageTransport` of the images to be processed.

**camera_info** A topic of `sensor_msgs/CameraInfo` messages with the camera
intrinsic parameters.


### aruco_detect aruco_detect

This node finds aruco markers in images stream and publishes their vertices
(corner points) and estimates 3D transforms from the camera to the fiducials.
It is based on the [Aruco](http://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html)
contributed module to OpenCV. It is an alternative to fiducial_detect

#### Parameters

**fiducial_len** The length of one side of a fiducial in meters, used by the
pose estimation.  Default 0.146.


#### Published Topics


**/fiducial_vertices** A topic of `fiducial_detect/Fiducial*` messages with the detected
fiducial vertices.


**/fiducial_transforms** A topic of `fiducial_pose/FiducialTransform` messages 
with the computed fiducial pose.

#### Subscribed Topics

**camera** An `ImageTransport` of the images to be processed.

**camera_info** A topic of `sensor_msgs/CameraInfo` messages with the camera
intrinsic parameters.

### fiducial_slam fiducial_slam.py

This node performs 3D Simultaneous Localization and Mapping (SLAM) from the 
fiducial transforms. For the mapping part, pairs of transforms are combined
to determine the position of fiducials based on existing observations.
For the localization part, fiducial transforms are combined with fiducial poses
to estimate the camera pose (and hence the robot pose).

#### Parameters

**map_file** Path to the file containing the generated map (this must exist). Default `map.txt`.

**trans_file** Path to a file to store all detected fiducial transforms. Default `trans.txt`.

**obs_file** Path to a file to store all detected fiducial observations. Default `obs.txt`.

**odom_frame** If this is set to a non-empty string, then the result of the localization is
published as a correction to odometry.  For example, the odometry publishes the tf from map
to odom, and this node publishes the tf from odom to base_link, with the tf from
map to odom removed. Default: not set.
 
**map_frame** The name of the map (world) frame.  Default `map`.

**pose_frame** The frame for our tf. Default `base_link`.

**publish_tf** If `true`, transforms are published. Default `true`.

**republish_tf** If `true`, transforms are republished until a new pose is calculated. Default `true`.

**mapping_mode** If `true` the map updated and saved more frequently.

**use_external_pose** If `true` then the node will attempt to use an external 
estimate of the robot pose (e.g. from AMCL) to estimate the pose of fiducials
if no known fiducials are observed.

**future** Amount of time (in seconds) to future-date published transforms.
Default 0.0.

**fiducials_are_level** If `true`, it is assumed that all fiducials are level, as
would be the case on ceiling mounted fiducials. In this case only 3DOF are estimated.
Default `true`.

#### Published Topics

**/fiducials** A topic of `visualization_msgs/Marker` messages that can be viewed
in rviz for debugging.

**/fiducial_pose** a topic of `geometry_msgs/PoseWithCovarianceStamped` containing
the computed pose.

**tf** Transforms


#### Subscribed Topics

**/fiducial_transforms** A topic of `fiducial_pose/FiducialTransform` messages with
fiducial pose.

**/tf** Transforms

### fiducial_slam init_amcl.py

This node will reinitialize amcl by republishing the pose reported from 
fiducial_slam.py to amcl as an initial pose.

#### Parameters

**cov_thresh** The threshold of covariance reported in *amcl_pose* for
reinitializing it.  Default 0.2.

#### Published Topics

**initial_pose** (geometry_msgs/PoseWithCovarianceStamped) The initial pose 
sent to AMCL

#### Subscribed Topics

**amcl_pose** (geometry_msgs/PoseWithCovarianceStamped) The pose from AMCL. The
covariance of this is examined to determine if AMCL needs re-initializing.

**fiducial_pose** (geometry_msgs/PoseWithCovarianceStamped) The pose from 
fiducial_slam.py

## File Formats:

### `map.txt` file format:

The format of `map.txt` is a series of lines of the form:

        id x y z pan tilt roll variance numObservations [neighbors ...]

