/**
MIT License

Copyright (c) 2020 Brennan Cain (Unmanned Systems and Robotics Lab,
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

#include <vector>
// #include <cv.hpp>
#include <opencv2/opencv.hpp>

namespace stag_ros {

struct Common {
  static void solvePnpSingle(const std::vector<cv::Point2d> &img,
                             const std::vector<cv::Point3d> &world,
                             cv::Mat &output, const cv::Mat &cameraMatrix,
                             const cv::Mat &distortionMatrix) {
    if (img.empty() or world.empty()) return;
    cv::Mat rVec, rMat, tVec;
    // optimize for 5 planar points
    // possibly choose to reduce to the 4 for use with advanced algos
    cv::solvePnP(world, img, cameraMatrix, distortionMatrix, rVec, tVec);
    cv::Rodrigues(rVec, rMat);
    rMat.convertTo(output.colRange(0, 3), CV_64F);
    tVec.convertTo(output.col(3), CV_64F);
  }

  static void solvePnpBundle(const std::vector<cv::Point2d> &img,
                             const std::vector<cv::Point3d> &world,
                             cv::Mat &output, const cv::Mat &cameraMatrix,
                             const cv::Mat &distortionMatrix) {
    if (img.empty() or world.empty()) return;
    cv::Mat rVec, rMat, tVec;
    // optimize for many points
    cv::solvePnP(world, img, cameraMatrix, distortionMatrix, rVec, tVec);
    cv::Rodrigues(rVec, rMat);
    rMat.convertTo(output.colRange(0, 3), CV_64F);
    tVec.convertTo(output.col(3), CV_64F);
  }

  static void publishTransform(
      const geometry_msgs::msg::Transform &transform,
      const std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::PoseStamped>> &publisher,
      const std_msgs::msg::Header &header, const std::string &frame_prefix,
      const std::string &frame_id, bool publish_tf, rclcpp::Node::SharedPtr node) {
    
    if (publish_tf) {
      static std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster =
        std::make_shared<tf2_ros::TransformBroadcaster>(node);

      geometry_msgs::msg::TransformStamped transform_stamped;
      transform_stamped.header = header;
      transform_stamped.child_frame_id = frame_prefix + frame_id;
      transform_stamped.transform = transform;
      tf_broadcaster->sendTransform(transform_stamped);
    }

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.frame_id = frame_id;
    pose_msg.header.stamp = header.stamp;
    pose_msg.pose.position.x = transform.translation.x;
    pose_msg.pose.position.y = transform.translation.y;
    pose_msg.pose.position.z = transform.translation.z;
    pose_msg.pose.orientation = transform.rotation;

    publisher->publish(pose_msg);
  }

  bool checkCoplanar(std::vector<cv::Point3d> worldP) {
    cv::Mat M = cv::Mat::zeros(3, worldP.size(), CV_64F);

    for (size_t i = 0; i < worldP.size(); ++i) {
      M.at<double>(0, i) = worldP[i].x;
      M.at<double>(1, i) = worldP[i].y;
      M.at<double>(2, i) = worldP[i].z;
    }

    cv::Mat w;
    cv::SVD::compute(M, w);
    cv::Mat nonZeroValues = w > 0.0001;
    int rank = countNonZero(nonZeroValues);

    if (rank > 2)
      return false;
    else
      return true;
  }

};  // namespace stag_ros
}  // namespace stag_ros