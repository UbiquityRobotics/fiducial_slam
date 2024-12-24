/**
MIT License

Copyright (c) 2020 Michail Kalaitzakis and Brennan Cain (Unmanned Systems and
Robotics Lab, University of South Carolina, USA)

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
#define NDEBUG
// Project includes
#ifndef NDEBUG
#include "stag_ros/instrument.hpp"
#endif

#include "stag_ros/stag_node.h"
#include "stag_ros/utility.hpp"
// Stag marker handle
#include "stag/Marker.h"

// ROS includes
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

#include <stdexcept>
#include <iostream>
#include <stag_ros/common.hpp>

namespace stag_ros {

StagNode::StagNode() : Node("stag_detect") {
  // Load Parameters
  loadParameters();

  // Initialize Stag
  try {
    stag = new Stag(stag_library, error_correction, false);
  } catch (const std::invalid_argument &e) {
    RCLCPP_ERROR(this->get_logger(), "%s", e.what());
    exit(-1);
  }

  // Set subscribers
  imageSub = image_transport::create_subscription(
      this, image_topic,
      std::bind(&StagNode::imageCallback, this, std::placeholders::_1),
      is_compressed ? "compressed" : "raw");


  cameraInfoSub = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic, 10,
      std::bind(&StagNode::cameraInfoCallback, this, std::placeholders::_1));

  // Set Publishers
  if (debug_images)
      imageDebugPub = image_transport::create_publisher(this, "stag_ros/image_markers");

  // imageDebugPub = imageT.advertise("stag_ros/image_markers", 1);
  markersPub = this->create_publisher<geometry_msgs::msg::PoseStamped>(markers_topic, 10);
  markersArrayPub = this->create_publisher<vision_msgs::msg::Detection2DArray>(markers_array_topic, 10);

  // Initialize camera info
  got_camera_info = false;

  cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
  distortionMat = cv::Mat::zeros(1, 5, CV_64F);
  rectificationMat = cv::Mat::zeros(3, 3, CV_64F);
  projectionMat = cv::Mat::zeros(3, 4, CV_64F);
}

StagNode::~StagNode() { delete stag; }

void StagNode::loadParameters() {
    // Declare and load parameters
    stag_library = this->declare_parameter<int>("stag_library", 15);
    error_correction = this->declare_parameter<int>("error_correction", 7);
    image_topic = this->declare_parameter<std::string>("image_topic", "image_raw");
    camera_info_topic = this->declare_parameter<std::string>("camera_info_topic", "camera_info");
    markers_topic = this->declare_parameter<std::string>("markers_topic", "stag_ros/markers");
    markers_array_topic = this->declare_parameter<std::string>("markers_array_topic", "stag_ros/markers_array");
    is_compressed = this->declare_parameter<bool>("is_compressed", false);
    debug_images = this->declare_parameter<bool>("debug_images", true);
    publish_tf = this->declare_parameter<bool>("publish_tf", false);
    tag_tf_prefix = this->declare_parameter<std::string>("tag_tf_prefix", "STag_");
    marker_size = this->declare_parameter<float>("marker_size", 0.18f);

}

void StagNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
#ifndef NDEBUG
  INSTRUMENT;
#endif
  if (got_camera_info) {
    // RCLCPP_INFO(this->get_logger(), "got image h:%d, w:%d, enc:%s", msg->height, msg->width, msg->encoding.c_str());

    cv::Mat gray;
    msgToGray(msg, gray);
    // RCLCPP_INFO(this->get_logger(), "gray image h:%d, w:%d", gray.rows, gray.cols);


    // Process the image to find the markers
    stag->detectMarkers(gray);
    std::vector<Marker> markers = stag->getMarkerList();

    // Publish debug image
    if (debug_images) {
      cv_bridge::CvImage rosMat;
      rosMat.header = msg->header;
      rosMat.encoding = "bgr8";
      rosMat.image = stag->drawMarkers();

      sensor_msgs::msg::Image rosImage;
      rosMat.toImageMsg(rosImage);

      imageDebugPub.publish(rosImage);
    }

    // For each marker in the list
    // if (markers.size() > 0) {
      // ROS_INFO("STag: Marker detected");

      vision_msgs::msg::Detection2DArray array;
      array.header = msg->header;

      for (int i = 0; i < markers.size(); i++) {

          std::vector<cv::Point2d> tag_image(5);
          std::vector<cv::Point3d> tag_world(5);

          tag_image[0] = markers[i].center;
          tag_world[0] = cv::Point3d(0.0, 0.0, 0.0);

          for (size_t ci = 0; ci < 4; ++ci) {
            tag_image[ci + 1] = markers[i].corners[ci];
          }

          float half_makrer_size = marker_size/2.0;
          // Top left
          tag_world[1] = cv::Point3d(-half_makrer_size, half_makrer_size, 0.0);
          // Top right
          tag_world[2] = cv::Point3d(half_makrer_size, half_makrer_size, 0.0);
          // Bottom right
          tag_world[3] = cv::Point3d(half_makrer_size, -half_makrer_size, 0.0);
          // Bottom left
          tag_world[4] = cv::Point3d(-half_makrer_size, -half_makrer_size, 0.0);


          cv::Mat marker_pose = cv::Mat::zeros(3, 4, CV_64F);
          Common::solvePnpSingle(tag_image, tag_world, marker_pose,
                                 cameraMatrix, distortionMat);

      if (marker_pose.empty()) return;

      tf2::Matrix3x3 rotMat(
        marker_pose.at<double>(0, 0), marker_pose.at<double>(0, 1),
        marker_pose.at<double>(0, 2), marker_pose.at<double>(1, 0),
        marker_pose.at<double>(1, 1), marker_pose.at<double>(1, 2),
        marker_pose.at<double>(2, 0), marker_pose.at<double>(2, 1),
        marker_pose.at<double>(2, 2));
      tf2::Quaternion rotQ;
      rotMat.getRotation(rotQ);

      tf2::Vector3 tfVec(marker_pose.at<double>(0, 3),
			marker_pose.at<double>(1, 3),
			marker_pose.at<double>(2, 3));

      // Create geometry_msgs::msg::Transform
      geometry_msgs::msg::Transform transform_msg;
      transform_msg.translation.x = tfVec.x();
      transform_msg.translation.y = tfVec.y();
      transform_msg.translation.z = tfVec.z();
      transform_msg.rotation.x = rotQ.x();
      transform_msg.rotation.y = rotQ.y();
      transform_msg.rotation.z = rotQ.z();
      transform_msg.rotation.w = rotQ.w();

      Common::publishTransform(transform_msg, markersPub, msg->header,
			       tag_tf_prefix, std::to_string(markers[i].id), publish_tf, shared_from_this());


      vision_msgs::msg::Detection2D markerobj;
      markerobj.header = msg->header;
      vision_msgs::msg::ObjectHypothesisWithPose marker;

      // Convert transform to pose
      geometry_msgs::msg::Pose pose;
      pose.position.x = transform_msg.translation.x;
      pose.position.y = transform_msg.translation.y;
      pose.position.z = transform_msg.translation.z;
      pose.orientation.x = transform_msg.rotation.x;
      pose.orientation.y = transform_msg.rotation.y;
      pose.orientation.z = transform_msg.rotation.z;
      pose.orientation.w = transform_msg.rotation.w;

      marker.pose.pose = pose;
      // marker.id = markers[i].id;
      marker.hypothesis.class_id = std::to_string(markers[i].id);
      markerobj.results.push_back(marker);

      array.detections.push_back(markerobj);
    }

    markersArrayPub->publish(array);
  // }
  /*
    else {
      ROS_WARN("No markers detected");
  }
  */
  }
}

void StagNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &msg) {
  // RCLCPP_INFO(this->get_logger(), "got camera info");
  // if (msg->k.size() < 9 || msg->d.size() < 5 || msg->r.size() < 9 || msg->p.size() < 12) {
  //     RCLCPP_INFO(this->get_logger(), "d size: %d, dist model: %s", msg->d.size(), msg->distortion_model.c_str());

  //     throw std::runtime_error("Insufficient data in camera info message");
  // }

  if (!got_camera_info) {
    // Get camera Matrix
    cameraMatrix.at<double>(0, 0) = msg->k[0];
    cameraMatrix.at<double>(0, 1) = msg->k[1];
    cameraMatrix.at<double>(0, 2) = msg->k[2];
    cameraMatrix.at<double>(1, 0) = msg->k[3];
    cameraMatrix.at<double>(1, 1) = msg->k[4];
    cameraMatrix.at<double>(1, 2) = msg->k[5];
    cameraMatrix.at<double>(2, 0) = msg->k[6];
    cameraMatrix.at<double>(2, 1) = msg->k[7];
    cameraMatrix.at<double>(2, 2) = msg->k[8];

    // Get distortion Matrix
    distortionMat.at<double>(0, 0) = msg->d[0];
    distortionMat.at<double>(0, 1) = msg->d[1];
    distortionMat.at<double>(0, 2) = msg->d[2];
    distortionMat.at<double>(0, 3) = msg->d[3];
    distortionMat.at<double>(0, 4) = msg->d[4];
    // Get rectification Matrix
    rectificationMat.at<double>(0, 0) = msg->r[0];
    rectificationMat.at<double>(0, 1) = msg->r[1];
    rectificationMat.at<double>(0, 2) = msg->r[2];
    rectificationMat.at<double>(1, 0) = msg->r[3];
    rectificationMat.at<double>(1, 1) = msg->r[4];
    rectificationMat.at<double>(1, 2) = msg->r[5];
    rectificationMat.at<double>(2, 0) = msg->r[6];
    rectificationMat.at<double>(2, 1) = msg->r[7];
    rectificationMat.at<double>(2, 2) = msg->r[8];
    // Get projection Matrix
    projectionMat.at<double>(0, 0) = msg->p[0];
    projectionMat.at<double>(0, 1) = msg->p[1];
    projectionMat.at<double>(0, 2) = msg->p[2];
    projectionMat.at<double>(1, 0) = msg->p[3];
    projectionMat.at<double>(1, 1) = msg->p[4];
    projectionMat.at<double>(1, 2) = msg->p[5];
    projectionMat.at<double>(2, 0) = msg->p[6];
    projectionMat.at<double>(2, 1) = msg->p[7];
    projectionMat.at<double>(2, 2) = msg->p[8];
    projectionMat.at<double>(2, 0) = msg->p[9];
    projectionMat.at<double>(2, 1) = msg->p[10];
    projectionMat.at<double>(2, 2) = msg->p[11];

    got_camera_info = true;
  }

}
}  // namespace stag_ros

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<stag_ros::StagNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}