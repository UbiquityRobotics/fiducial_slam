#pragma once

namespace stag_ros {

// Read image from msg and convert it to grayscale, checks provided for rgb8
// and bgr8, default to mono8
inline bool msgToGray(const sensor_msgs::msg::Image::ConstSharedPtr &msg, cv::Mat &gray) {
  if (msg->encoding == "yuv422_yuy2") {
    // Convert raw YUV422 data to a cv::Mat
    cv::Mat yuv_image(msg->height, msg->width, CV_8UC2, const_cast<uchar*>(msg->data.data()));

    // Convert YUV422 to GRAY
    cv::cvtColor(yuv_image, gray, cv::COLOR_YUV2GRAY_YUY2);
    return true;
  } else if (msg->encoding.compare("bgr8") == 0) {
    cv::Mat src = cv_bridge::toCvShare(msg, msg->encoding)->image;
    cv::cvtColor(src, gray, CV_BGR2GRAY);
    return true;
  } else if (msg->encoding.compare("rgb8") == 0) {
    cv::Mat src = cv_bridge::toCvShare(msg, msg->encoding)->image;
    cv::cvtColor(src, gray, CV_RGB2GRAY);
    return true;
  } else if (msg->encoding.compare("mono8") == 0) {
    gray = cv_bridge::toCvShare(msg, msg->encoding)->image;
    return true;
  }
  return false;
}

}  // namespace stag_ros