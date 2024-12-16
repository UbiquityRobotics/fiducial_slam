/**
MIT License

Copyright (c) 2020 Michail Kalaitzakis (Unmanned Systems and Robotics Lab,
University of South Carolina, USA)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

// ROS includes
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "image_transport/image_transport.hpp"
#include "camera_info_manager/camera_info_manager.hpp"

// ROS msgs
#include "sensor_msgs/image_encodings.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection2_d.hpp"


// Stag includes
#include "stag/Stag.h"
#include "stag_ros/structures.hpp"

namespace stag_ros {
class StagNode : public rclcpp::Node {
 public:
  StagNode();
  ~StagNode();

 private:
  // Callbacks
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  // void markersArrayCallback(const geometry_msgs::PoseStampedConstPtr &msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &msg);

  // Functions
  void loadParameters();
  // bool getTagIndex(const int id, int &tag_index);

  // STag handle
  Stag *stag;
  int stag_library;
  int error_correction;
  float marker_size;

  // ROS 2 Interfaces
  image_transport::Subscriber imageSub;
  image_transport::Publisher imageDebugPub;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoSub;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr markersPub;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr markersArrayPub;

  // Data
  cv::Mat cameraMatrix;
  cv::Mat distortionMat;
  cv::Mat rectificationMat;
  cv::Mat projectionMat;
  bool got_camera_info;
  bool debug_images;
  bool publish_tf;
  bool is_compressed;
  std::string image_topic;
  std::string camera_info_topic;
  std::string markers_topic;
  std::string markers_array_topic;
  std::string tag_tf_prefix;

  // Tag and bundle info
  std::map<std::string, vision_msgs::msg::Detection2D> markersFrames;
  //std::vector<Bundle> bundles;
  //std::vector<Tag> tags;
  
};

}  // namespace stag_ros
