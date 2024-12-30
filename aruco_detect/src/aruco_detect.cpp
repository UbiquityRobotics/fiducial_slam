/*
 * Copyright (c) 2017-20, Ubiquity Robotics Inc., Austin Hendrix
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the FreeBSD Project.
 *
 */

#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>

#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "fiducial_msgs/msg/fiducial.hpp"
#include "fiducial_msgs/msg/fiducial_array.hpp"
#include "fiducial_msgs/msg/fiducial_transform.hpp"
#include "fiducial_msgs/msg/fiducial_transform_array.hpp"

#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include <list>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace cv;

typedef std::shared_ptr< fiducial_msgs::msg::FiducialArray const> FiducialArrayConstPtr;

class FiducialsNode : public rclcpp::Node {
  private:


    // Subscribers and publishers
    image_transport::Subscriber img_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr ignore_sub;
    rclcpp::Subscription<fiducial_msgs::msg::FiducialArray>::SharedPtr vertices_sub;

    rclcpp::Publisher<fiducial_msgs::msg::FiducialArray>::SharedPtr vertices_pub;
    rclcpp::Publisher<fiducial_msgs::msg::FiducialTransformArray>::SharedPtr pose_pub_fta;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pose_pub_d2a;
    image_transport::Publisher image_pub;

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr service_enable_detections;

    tf2_ros::TransformBroadcaster broadcaster;

    // if set, we publish the images that contain fiducials
    bool publish_images;
    bool enable_detections;
    bool vis_msgs;
    bool verbose;

    double fiducial_len;

    bool doPoseEstimation;
    bool haveCamInfo;
    bool publishFiducialTf;
    vector <vector <Point2f> > corners;
    vector <int> ids;
    cv_bridge::CvImagePtr cv_ptr;

    cv::Mat cameraMatrix;
    cv::Mat distortionCoeffs;
    int frameNum;
    std::string frameId;
    std::vector<int> ignoreIds;
    std::map<int, double> fiducialLens;

    // log spam prevention
    int prev_detected_count;

    cv::Ptr<aruco::DetectorParameters> detectorParams;
    cv::Ptr<aruco::Dictionary> dictionary;

    void handleIgnoreString(const std::string& str);

    void estimatePoseSingleMarkers(float markerLength,
                                   const cv::Mat &cameraMatrix,
                                   const cv::Mat &distCoeffs,
                                   vector<Vec3d>& rvecs, vector<Vec3d>& tvecs,
                                   vector<double>& reprojectionError);


    void ignoreCallback(const std_msgs::msg::String &msg);
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
    void poseEstimateCallback(const FiducialArrayConstPtr &msg);
    void camInfoCallback(const std::shared_ptr<const sensor_msgs::msg::CameraInfo> msg);
    // void configCallback(aruco_detect::DetectorParamsConfig &config, uint32_t level);

    bool enableDetectionsCallback(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);

    // dynamic_reconfigure::Server<aruco_detect::DetectorParamsConfig> configServer;
    // dynamic_reconfigure::Server<aruco_detect::DetectorParamsConfig>::CallbackType callbackType;

  public:
    FiducialsNode();

    void init();
};


/**
  * @brief Return object points for the system centered in a single marker, given the marker length
  */
static void getSingleMarkerObjectPoints(float markerLength, vector<Point3f>& objPoints) {

    CV_Assert(markerLength > 0);

    // set coordinate system in the middle of the marker, with Z pointing out
    objPoints.clear();
    objPoints.push_back(Vec3f(-markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3f( markerLength / 2.f, markerLength / 2.f, 0));
    objPoints.push_back(Vec3f( markerLength / 2.f,-markerLength / 2.f, 0));
    objPoints.push_back(Vec3f(-markerLength / 2.f,-markerLength / 2.f, 0));
}

// Euclidean distance between two points
static double dist(const cv::Point2f &p1, const cv::Point2f &p2)
{
    double x1 = p1.x;
    double y1 = p1.y;
    double x2 = p2.x;
    double y2 = p2.y;

    double dx = x1 - x2;
    double dy = y1 - y2;

    return sqrt(dx*dx + dy*dy);
}

// Compute area in image of a fiducial, using Heron's formula
// to find the area of two triangles
static double calcFiducialArea(const std::vector<cv::Point2f> &pts)
{
    const Point2f &p0 = pts.at(0);
    const Point2f &p1 = pts.at(1);
    const Point2f &p2 = pts.at(2);
    const Point2f &p3 = pts.at(3);

    double a1 = dist(p0, p1);
    double b1 = dist(p0, p3);
    double c1 = dist(p1, p3);

    double a2 = dist(p1, p2);
    double b2 = dist(p2, p3);
    double c2 = c1;

    double s1 = (a1 + b1 + c1) / 2.0;
    double s2 = (a2 + b2 + c2) / 2.0;

    a1 = sqrt(s1*(s1-a1)*(s1-b1)*(s1-c1));
    a2 = sqrt(s2*(s2-a2)*(s2-b2)*(s2-c2));
    return a1+a2;
}

// estimate reprojection error
static double getReprojectionError(const vector<Point3f> &objectPoints,
                            const vector<Point2f> &imagePoints,
                            const Mat &cameraMatrix, const Mat  &distCoeffs,
                            const Vec3d &rvec, const Vec3d &tvec) {

    vector<Point2f> projectedPoints;

    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix,
                      distCoeffs, projectedPoints);

    // calculate RMS image error
    double totalError = 0.0;
    for (unsigned int i=0; i<objectPoints.size(); i++) {
        double error = dist(imagePoints[i], projectedPoints[i]);
        totalError += error*error;
    }
    double rerror = totalError/(double)objectPoints.size();
    return rerror;
}

void FiducialsNode::estimatePoseSingleMarkers(float markerLength,
                                const cv::Mat &cameraMatrix,
                                const cv::Mat &distCoeffs,
                                vector<Vec3d>& rvecs, vector<Vec3d>& tvecs,
                                vector<double>& reprojectionError) {

    CV_Assert(markerLength > 0);

    vector<Point3f> markerObjPoints;
    int nMarkers = (int)corners.size();
    rvecs.reserve(nMarkers);
    tvecs.reserve(nMarkers);
    reprojectionError.reserve(nMarkers);

    // for each marker, calculate its pose
    for (int i = 0; i < nMarkers; i++) {
       double fiducialSize = markerLength;

       std::map<int, double>::iterator it = fiducialLens.find(ids[i]);
       if (it != fiducialLens.end()) {
          fiducialSize = it->second;
       }

       getSingleMarkerObjectPoints(fiducialSize, markerObjPoints);
       cv::solvePnP(markerObjPoints, corners[i], cameraMatrix, distCoeffs,
                    rvecs[i], tvecs[i]);

       reprojectionError[i] =
          getReprojectionError(markerObjPoints, corners[i],
                               cameraMatrix, distCoeffs,
                               rvecs[i], tvecs[i]);
    }
}

// void FiducialsNode::configCallback(aruco_detect::DetectorParamsConfig & config, uint32_t level)
// {
//     /* Don't load initial config, since it will overwrite the rosparam settings */
//     if (level == 0xFFFFFFFF) {
//         return;
//     }

//     detectorParams->adaptiveThreshConstant = config.adaptiveThreshConstant;
//     detectorParams->adaptiveThreshWinSizeMin = config.adaptiveThreshWinSizeMin;
//     detectorParams->adaptiveThreshWinSizeMax = config.adaptiveThreshWinSizeMax;
//     detectorParams->adaptiveThreshWinSizeStep = config.adaptiveThreshWinSizeStep;
//     detectorParams->cornerRefinementMaxIterations = config.cornerRefinementMaxIterations;
//     detectorParams->cornerRefinementMinAccuracy = config.cornerRefinementMinAccuracy;
//     detectorParams->cornerRefinementWinSize = config.cornerRefinementWinSize;
// #if CV_MINOR_VERSION==2 and CV_MAJOR_VERSION==3
//     detectorParams->doCornerRefinement = config.doCornerRefinement;
// #else
//     if (config.doCornerRefinement) {
//        if (config.cornerRefinementSubpix) {
//          detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
//        }
//        else {
//          detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;
//        }
//     }
//     else {
//        detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_NONE;
//     }
// #endif
//     detectorParams->errorCorrectionRate = config.errorCorrectionRate;
//     detectorParams->minCornerDistanceRate = config.minCornerDistanceRate;
//     detectorParams->markerBorderBits = config.markerBorderBits;
//     detectorParams->maxErroneousBitsInBorderRate = config.maxErroneousBitsInBorderRate;
//     detectorParams->minDistanceToBorder = config.minDistanceToBorder;
//     detectorParams->minMarkerDistanceRate = config.minMarkerDistanceRate;
//     detectorParams->minMarkerPerimeterRate = config.minMarkerPerimeterRate;
//     detectorParams->maxMarkerPerimeterRate = config.maxMarkerPerimeterRate;
//     detectorParams->minOtsuStdDev = config.minOtsuStdDev;
//     detectorParams->perspectiveRemoveIgnoredMarginPerCell = config.perspectiveRemoveIgnoredMarginPerCell;
//     detectorParams->perspectiveRemovePixelPerCell = config.perspectiveRemovePixelPerCell;
//     detectorParams->polygonalApproxAccuracyRate = config.polygonalApproxAccuracyRate;
// }

void FiducialsNode::ignoreCallback(const std_msgs::msg::String& msg)
{
    ignoreIds.clear();
    // pnh.setParam("ignore_fiducials", msg.data);
    this->set_parameter(rclcpp::Parameter("ignore_fiducials", msg.data));

    handleIgnoreString(msg.data);
}

void FiducialsNode::camInfoCallback(const std::shared_ptr<const sensor_msgs::msg::CameraInfo> msg)
{
    if (haveCamInfo) {
        return;
    }

    if (msg->k != std::array<double, 9>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}) {
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                cameraMatrix.at<double>(i, j) = msg->k[i*3+j];
            }
        }

        for (int i=0; i<5; i++) {
            distortionCoeffs.at<double>(0,i) = msg->d[i];
        }

        haveCamInfo = true;
        frameId = msg->header.frame_id;
    }
    else {
        RCLCPP_WARN(this->get_logger(), "%s", "CameraInfo message has invalid intrinsics, K matrix all zeros");
    }
}

void FiducialsNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
    if (enable_detections == false) {
        return; //return without doing anything
    }

    if(verbose){
        // RCLCPP_INFO(this->get_logger(), "Got image %d", msg->header.seq);       
        RCLCPP_INFO(this->get_logger(), "Got image with timestamp: %u.%u", msg->header.stamp.sec, msg->header.stamp.nanosec);

    }

    fiducial_msgs::msg::FiducialArray fva;
    fva.header.stamp = msg->header.stamp;
    fva.header.frame_id = frameId;
    // fva.image_seq = msg->header.seq;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        aruco::detectMarkers(cv_ptr->image, dictionary, corners, ids, detectorParams);

        int detected_count = (int)ids.size();
        if(verbose || detected_count != prev_detected_count){
            prev_detected_count = detected_count;
            RCLCPP_INFO(this->get_logger(), "Detected %d markers", detected_count);
        }

        for (size_t i=0; i<ids.size(); i++) {
            if (std::count(ignoreIds.begin(), ignoreIds.end(), ids[i]) != 0) {
                if(verbose){
                    RCLCPP_INFO(this->get_logger(), "Ignoring id %d", ids[i]);                    
                }
                continue;
            }
            fiducial_msgs::msg::Fiducial fid;
            fid.fiducial_id = ids[i];

            fid.x0 = corners[i][0].x;
            fid.y0 = corners[i][0].y;
            fid.x1 = corners[i][1].x;
            fid.y1 = corners[i][1].y;
            fid.x2 = corners[i][2].x;
            fid.y2 = corners[i][2].y;
            fid.x3 = corners[i][3].x;
            fid.y3 = corners[i][3].y;
            fva.fiducials.push_back(fid);
        }

        vertices_pub->publish(fva);

        if(ids.size() > 0) {
            aruco::drawDetectedMarkers(cv_ptr->image, corners, ids);
        }

        if (publish_images) {
            image_pub.publish(cv_ptr->toImageMsg());
        }
    }
    catch(cv_bridge::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
    catch(cv::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "cv exception: %s", e.what());
    }
}

void FiducialsNode::poseEstimateCallback(const FiducialArrayConstPtr & msg)
{
    vector <Vec3d>  rvecs, tvecs;

    vision_msgs::msg::Detection2DArray vma;
    fiducial_msgs::msg::FiducialTransformArray fta;
    if (vis_msgs) {
        vma.header.stamp = msg->header.stamp;
        vma.header.frame_id = frameId;
        // vma.header.seq = msg->header.seq;
    }
    else {
        fta.header.stamp = msg->header.stamp;
        fta.header.frame_id = frameId;
        // fta.image_seq = msg->header.seq;
    }
    frameNum++;

    if (doPoseEstimation) {
        try {
            if (!haveCamInfo) {
                if (frameNum > 5) {
                    RCLCPP_ERROR(this->get_logger(), "No camera intrinsics");
                }
                return;
            }

            vector <double>reprojectionError;
            estimatePoseSingleMarkers((float)fiducial_len,
                                      cameraMatrix, distortionCoeffs,
                                      rvecs, tvecs,
                                      reprojectionError);

            for (size_t i=0; i<ids.size(); i++) {
                drawFrameAxes(cv_ptr->image, cameraMatrix, distortionCoeffs,
                                rvecs[i], tvecs[i], (float)fiducial_len);
                if(verbose){
                    RCLCPP_INFO(this->get_logger(), "Detected id %d T %.2f %.2f %.2f R %.2f %.2f %.2f", ids[i],
                         tvecs[i][0], tvecs[i][1], tvecs[i][2],
                         rvecs[i][0], rvecs[i][1], rvecs[i][2]);

                }

                if (std::count(ignoreIds.begin(), ignoreIds.end(), ids[i]) != 0) {
                    if(verbose){
                        RCLCPP_INFO(this->get_logger(), "Ignoring id %d", ids[i]);
                    }
                    continue;
                }

                double angle = norm(rvecs[i]);
                Vec3d axis = rvecs[i] / angle;

                if(verbose){
                    RCLCPP_INFO(this->get_logger(), "angle %f axis %f %f %f",
                         angle, axis[0], axis[1], axis[2]);
                }

                double object_error =
                        (reprojectionError[i] / dist(corners[i][0], corners[i][2])) *
                        (norm(tvecs[i]) / fiducial_len);

                // Standard ROS vision_msgs
                fiducial_msgs::msg::FiducialTransform ft;
                tf2::Quaternion q;
                if (vis_msgs) {
                    vision_msgs::msg::Detection2D vm;
                    vision_msgs::msg::ObjectHypothesisWithPose vmh;
                    vmh.hypothesis.class_id = ids[i];
                    vmh.hypothesis.score = exp(-2 * object_error); // [0, infinity] -> [1,0]
                    vmh.pose.pose.position.x = tvecs[i][0];
                    vmh.pose.pose.position.y = tvecs[i][1];
                    vmh.pose.pose.position.z = tvecs[i][2];
                    q.setRotation(tf2::Vector3(axis[0], axis[1], axis[2]), angle);
                    vmh.pose.pose.orientation.w = q.w();
                    vmh.pose.pose.orientation.x = q.x();
                    vmh.pose.pose.orientation.y = q.y();
                    vmh.pose.pose.orientation.z = q.z();

                    vm.results.push_back(vmh);
                    vma.detections.push_back(vm);
                }
                else {
                    ft.fiducial_id = ids[i];

                    ft.transform.translation.x = tvecs[i][0];
                    ft.transform.translation.y = tvecs[i][1];
                    ft.transform.translation.z = tvecs[i][2];
                    q.setRotation(tf2::Vector3(axis[0], axis[1], axis[2]), angle);
                    ft.transform.rotation.w = q.w();
                    ft.transform.rotation.x = q.x();
                    ft.transform.rotation.y = q.y();
                    ft.transform.rotation.z = q.z();
                    ft.fiducial_area = calcFiducialArea(corners[i]);
                    ft.image_error = reprojectionError[i];
                    // Convert image_error (in pixels) to object_error (in meters)
                    ft.object_error =
                        (reprojectionError[i] / dist(corners[i][0], corners[i][2])) *
                        (norm(tvecs[i]) / fiducial_len);

                    fta.transforms.push_back(ft);
                }

                // Publish tf for the fiducial relative to the camera
                if (publishFiducialTf) {
                    if (vis_msgs) {
                        geometry_msgs::msg::TransformStamped ts;
                        ts.transform.translation.x = tvecs[i][0];
                        ts.transform.translation.y = tvecs[i][1];
                        ts.transform.translation.z = tvecs[i][2];
                        ts.transform.rotation.w = q.w();
                        ts.transform.rotation.x = q.x();
                        ts.transform.rotation.y = q.y();
                        ts.transform.rotation.z = q.z();
                        ts.header.frame_id = frameId;
                        ts.header.stamp = msg->header.stamp;
                        ts.child_frame_id = "fiducial_" + std::to_string(ids[i]);
                        broadcaster.sendTransform(ts);
                    }
                    else {
                        geometry_msgs::msg::TransformStamped ts;
                        ts.transform = ft.transform;
                        ts.header.frame_id = frameId;
                        ts.header.stamp = msg->header.stamp;
                        ts.child_frame_id = "fiducial_" + std::to_string(ft.fiducial_id);
                        broadcaster.sendTransform(ts);
                    }
                }
            }
        }
        catch(cv_bridge::Exception & e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
        catch(cv::Exception & e) {
            RCLCPP_ERROR(this->get_logger(), "cv exception: %s", e.what());
        }
    }
    if (vis_msgs)
        pose_pub_d2a->publish(vma);
    else 
        pose_pub_fta->publish(fta);
}

void FiducialsNode::handleIgnoreString(const std::string& str)
{
    /*
    ignogre fiducials can take comma separated list of individual
    fiducial ids or ranges, eg "1,4,8,9-12,30-40"
    */
    std::vector<std::string> strs;
    boost::split(strs, str, boost::is_any_of(","));
    for (const string& element : strs) {
        if (element == "") {
           continue;
        }
        std::vector<std::string> range;
        boost::split(range, element, boost::is_any_of("-"));
        if (range.size() == 2) {
           int start = std::stoi(range[0]);
           int end = std::stoi(range[1]);
           RCLCPP_INFO(this->get_logger(), "Ignoring fiducial id range %d to %d", start, end);
           for (int j=start; j<=end; j++) {
               ignoreIds.push_back(j);
           }
        }
        else if (range.size() == 1) {
           int fid = std::stoi(range[0]);
           RCLCPP_INFO(this->get_logger(), "Ignoring fiducial id %d", fid);
           ignoreIds.push_back(fid);
        }
        else {
           RCLCPP_ERROR(this->get_logger(), "Malformed ignore_fiducials: %s", element.c_str());
        }
    }
}

bool FiducialsNode::enableDetectionsCallback(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
    enable_detections = request->data;
    if (enable_detections){
        response->message = "Enabled aruco detections.";
        RCLCPP_INFO(this->get_logger(), "Enabled aruco detections.");
    }
    else {
        response->message = "Disabled aruco detections.";
        RCLCPP_INFO(this->get_logger(), "Disabled aruco detections.");
    }

    response->success = true;
    return true;
}


FiducialsNode::FiducialsNode() : Node("aruco_detect"), broadcaster(this)
{
    frameNum = 0;
    prev_detected_count = -1;

    // Camera intrinsics
    cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);

    // distortion coefficients
    distortionCoeffs = cv::Mat::zeros(1, 5, CV_64F);

    haveCamInfo = false;
    enable_detections = true;

    int dicno;

    detectorParams = new aruco::DetectorParameters();

    this->declare_parameter("publish_images", false);
    this->declare_parameter("fiducial_len", 0.14);
    this->declare_parameter("dictionary", 7);
    this->declare_parameter("do_pose_estimation", true);
    this->declare_parameter("publish_fiducial_tf", true);
    this->declare_parameter("vis_msgs", false);
    this->declare_parameter("verbose", false);
    this->declare_parameter("ignore_fiducials", "");
    this->declare_parameter("fiducial_len_override", "");
    this->declare_parameter("image_transport", "raw");


    this->get_parameter("publish_images", publish_images);
    this->get_parameter("fiducial_len", fiducial_len);
    this->get_parameter("do_pose_estimation", doPoseEstimation);
    this->get_parameter("publish_fiducial_tf", publishFiducialTf);
    this->get_parameter("vis_msgs", vis_msgs);
    this->get_parameter("verbose", verbose);
    this->get_parameter("dictionary", dicno);


    std::string str;
    std::vector<std::string> strs;

    this->get_parameter("ignore_fiducials", str);

    // pnh.param<string>("ignore_fiducials", str, "");
    handleIgnoreString(str);

    /*
    fiducial size can take comma separated list of size: id or size: range,
    e.g. "200.0: 12, 300.0: 200-300"
    */
    this->get_parameter("fiducial_len_override", str);
    boost::split(strs, str, boost::is_any_of(","));
    for (const string& element : strs) {
        if (element == "") {
           continue;
        }
        std::vector<std::string> parts;
        boost::split(parts, element, boost::is_any_of(":"));
        if (parts.size() == 2) {
            double len = std::stod(parts[1]);
            std::vector<std::string> range;
            boost::split(range, element, boost::is_any_of("-"));
            if (range.size() == 2) {
               int start = std::stoi(range[0]);
               int end = std::stoi(range[1]);
               RCLCPP_INFO(this->get_logger(), "Setting fiducial id range %d - %d length to %f",
                        start, end, len);
               for (int j=start; j<=end; j++) {
                   fiducialLens[j] = len;
               }
            }
            else if (range.size() == 1){
               int fid = std::stoi(range[0]);
               RCLCPP_INFO(this->get_logger(), "Setting fiducial id %d length to %f", fid, len);
               fiducialLens[fid] = len;
            }
            else {
               RCLCPP_ERROR(this->get_logger(), "Malformed fiducial_len_override: %s", element.c_str());
            }
        }
        else {
           RCLCPP_ERROR(this->get_logger(), "Malformed fiducial_len_override: %s", element.c_str());
        }
    }

    vertices_pub = this->create_publisher<fiducial_msgs::msg::FiducialArray>("fiducial_vertices", 10);

    if (vis_msgs) {
        pose_pub_d2a = this->create_publisher<vision_msgs::msg::Detection2DArray>("fiducial_transforms", 10);
    } else {
        pose_pub_fta = this->create_publisher<fiducial_msgs::msg::FiducialTransformArray>("fiducial_transforms", 10);
    }

    image_pub = image_transport::create_publisher(this, "fiducial_images");


    dictionary = aruco::getPredefinedDictionary(dicno);

    vertices_sub = this->create_subscription<fiducial_msgs::msg::FiducialArray>(
        "fiducial_vertices", 10, std::bind(&FiducialsNode::poseEstimateCallback, this, std::placeholders::_1));


    caminfo_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", 10, std::bind(&FiducialsNode::camInfoCallback, this, std::placeholders::_1));

    ignore_sub = this->create_subscription<std_msgs::msg::String>(
        "ignore_fiducials", 10, std::bind(&FiducialsNode::ignoreCallback, this, std::placeholders::_1));


    service_enable_detections = this->create_service<std_srvs::srv::SetBool>("enable_detections",
            std::bind(&FiducialsNode::enableDetectionsCallback, this, std::placeholders::_1, std::placeholders::_2));


    // callbackType = boost::bind(&FiducialsNode::configCallback, this, boost::placeholders::_1, boost::placeholders::_2);
    // configServer.setCallback(callbackType);

    // Declare parameters with default values
    this->declare_parameter<double>("adaptiveThreshConstant", 7.0);
    this->declare_parameter<int>("adaptiveThreshWinSizeMax", 53);
    this->declare_parameter<int>("adaptiveThreshWinSizeMin", 3);
    this->declare_parameter<int>("adaptiveThreshWinSizeStep", 4);
    this->declare_parameter<int>("cornerRefinementMaxIterations", 30);
    this->declare_parameter<double>("cornerRefinementMinAccuracy", 0.01);
    this->declare_parameter<int>("cornerRefinementWinSize", 5);
    this->declare_parameter<bool>("doCornerRefinement", true);
    this->declare_parameter<bool>("cornerRefinementSubPix", true);
    this->declare_parameter<double>("errorCorrectionRate", 0.6);
    this->declare_parameter<double>("minCornerDistanceRate", 0.05);
    this->declare_parameter<int>("markerBorderBits", 1);
    this->declare_parameter<double>("maxErroneousBitsInBorderRate", 0.04);
    this->declare_parameter<int>("minDistanceToBorder", 3);
    this->declare_parameter<double>("minMarkerDistanceRate", 0.05);
    this->declare_parameter<double>("minMarkerPerimeterRate", 0.1);
    this->declare_parameter<double>("maxMarkerPerimeterRate", 4.0);
    this->declare_parameter<double>("minOtsuStdDev", 5.0);
    this->declare_parameter<double>("perspectiveRemoveIgnoredMarginPerCell", 0.13);
    this->declare_parameter<int>("perspectiveRemovePixelPerCell", 8);
    this->declare_parameter<double>("polygonalApproxAccuracyRate", 0.01);

    // Retrieve parameters
    detectorParams->adaptiveThreshConstant = this->get_parameter("adaptiveThreshConstant").as_double();
    detectorParams->adaptiveThreshWinSizeMax = this->get_parameter("adaptiveThreshWinSizeMax").as_int();
    detectorParams->adaptiveThreshWinSizeMin = this->get_parameter("adaptiveThreshWinSizeMin").as_int();
    detectorParams->adaptiveThreshWinSizeStep = this->get_parameter("adaptiveThreshWinSizeStep").as_int();
    detectorParams->cornerRefinementMaxIterations = this->get_parameter("cornerRefinementMaxIterations").as_int();
    detectorParams->cornerRefinementMinAccuracy = this->get_parameter("cornerRefinementMinAccuracy").as_double();
    detectorParams->cornerRefinementWinSize = this->get_parameter("cornerRefinementWinSize").as_int();

    bool doCornerRefinement = this->get_parameter("doCornerRefinement").as_bool();
    if (doCornerRefinement) {
        bool cornerRefinementSubPix = this->get_parameter("cornerRefinementSubPix").as_bool();
        detectorParams->cornerRefinementMethod = cornerRefinementSubPix
                                                  ? aruco::CORNER_REFINE_SUBPIX
                                                  : aruco::CORNER_REFINE_CONTOUR;
    } else {
        detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_NONE;
    }

    detectorParams->errorCorrectionRate = this->get_parameter("errorCorrectionRate").as_double();
    detectorParams->minCornerDistanceRate = this->get_parameter("minCornerDistanceRate").as_double();
    detectorParams->markerBorderBits = this->get_parameter("markerBorderBits").as_int();
    detectorParams->maxErroneousBitsInBorderRate = this->get_parameter("maxErroneousBitsInBorderRate").as_double();
    detectorParams->minDistanceToBorder = this->get_parameter("minDistanceToBorder").as_int();
    detectorParams->minMarkerDistanceRate = this->get_parameter("minMarkerDistanceRate").as_double();
    detectorParams->minMarkerPerimeterRate = this->get_parameter("minMarkerPerimeterRate").as_double();
    detectorParams->maxMarkerPerimeterRate = this->get_parameter("maxMarkerPerimeterRate").as_double();
    detectorParams->minOtsuStdDev = this->get_parameter("minOtsuStdDev").as_double();
    detectorParams->perspectiveRemoveIgnoredMarginPerCell = this->get_parameter("perspectiveRemoveIgnoredMarginPerCell").as_double();
    detectorParams->perspectiveRemovePixelPerCell = this->get_parameter("perspectiveRemovePixelPerCell").as_int();
    detectorParams->polygonalApproxAccuracyRate = this->get_parameter("polygonalApproxAccuracyRate").as_double();

    RCLCPP_INFO(this->get_logger(), "Aruco detection ready");
}

void FiducialsNode::init(){
    std::string transport;
    this->get_parameter("image_transport", transport);
    auto transport_hints = std::make_shared<image_transport::TransportHints>(this, transport);        // Define subscription options
    rclcpp::SubscriptionOptions options;
    rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
    // Use image_transport with the specified transport type
    image_transport::ImageTransport it(this->shared_from_this());
    img_sub = it.subscribe(
        "camera/image_raw",                                    // Base topic
        qos_profile.get_rmw_qos_profile(),                                                    // Queue size
        &FiducialsNode::imageCallback,                         // Member function pointer (fp)
        this,                                                  // Instance pointer (obj)
        transport_hints.get(),         // Transport hints
        options                          // Subscription options
    );

    RCLCPP_INFO(this->get_logger(), "Image transport initialized with '%s' transport", transport.c_str());

}


int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<FiducialsNode>();
    node->init();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
