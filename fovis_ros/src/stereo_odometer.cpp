#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <fovis_ros/FovisInfo.h>

#include <fovis/visual_odometry.hpp>
#include <fovis/stereo_depth.hpp>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>


#include "stereo_processor.hpp"
#include "visualization.hpp"

namespace fovis_ros
{

class StereoOdometer : public StereoProcessor
{

private:

  boost::shared_ptr<fovis::VisualOdometry> visual_odometer_;
  fovis::VisualOdometryOptions visual_odometer_options_;
  boost::shared_ptr<fovis::StereoDepth> stereo_depth_;

  ros::Time last_time_;

  // tf related
  std::string sensor_frame_id_;
  std::string odom_frame_id_;
  std::string base_link_frame_id_;
  bool publish_tf_;
  tf::StampedTransform initial_base_to_sensor_;
  tf::TransformListener tf_listener_;
  tf::TransformBroadcaster tf_broadcaster_;

  ros::NodeHandle nh_local_;

  // publisher
  ros::Publisher odom_pub_;
  ros::Publisher pose_pub_;
  ros::Publisher info_pub_;
  image_transport::Publisher features_pub_;
  image_transport::ImageTransport it_;

public:

  StereoOdometer(const std::string& transport) : 
    StereoProcessor(transport), 
    visual_odometer_options_(fovis::VisualOdometry::getDefaultOptions()),
    nh_local_("~"),
    it_(nh_local_)
  {
    // TODO load parameters from node handle to visual_odometer_options_

    nh_local_.param("odom_frame_id", odom_frame_id_, std::string("/odom"));
    nh_local_.param("base_link_frame_id", base_link_frame_id_, std::string("/base_link"));
    nh_local_.param("publish_tf", publish_tf_, true);

    odom_pub_ = nh_local_.advertise<nav_msgs::Odometry>("odometry", 1);
    pose_pub_ = nh_local_.advertise<geometry_msgs::PoseStamped>("pose", 1);
    info_pub_ = nh_local_.advertise<FovisInfo>("info", 1);
    features_pub_ = it_.advertise("features", 1);
  }

protected:

  void rosToFovis(const image_geometry::PinholeCameraModel& camera_model,
      fovis::CameraIntrinsicsParameters& parameters)
  {
    parameters.cx = camera_model.cx();
    parameters.cy = camera_model.cy();
    parameters.fx = camera_model.fx();
    parameters.fy = camera_model.fy();
  }

  void initOdometer(
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg)
  {
    // read calibration info from camera info message
    // to fill remaining parameters
    image_geometry::StereoCameraModel model;
    model.fromCameraInfo(*l_info_msg, *r_info_msg);
    
    // initialize left camera parameters
    fovis::CameraIntrinsicsParameters left_parameters;
    rosToFovis(model.left(), left_parameters);
    left_parameters.height = l_info_msg->height;
    left_parameters.width = l_info_msg->width;
    // initialize right camera parameters
    fovis::CameraIntrinsicsParameters right_parameters;
    rosToFovis(model.right(), right_parameters);
    right_parameters.height = r_info_msg->height;
    right_parameters.width = r_info_msg->width;

    // as we use rectified images, rotation is identity
    // and translation is baseline only
    fovis::StereoCalibrationParameters stereo_parameters;
    stereo_parameters.left_parameters = left_parameters;
    stereo_parameters.right_parameters = right_parameters;
    stereo_parameters.right_to_left_rotation[0] = 1.0;
    stereo_parameters.right_to_left_rotation[1] = 0.0;
    stereo_parameters.right_to_left_rotation[2] = 0.0;
    stereo_parameters.right_to_left_rotation[3] = 0.0;
    stereo_parameters.right_to_left_translation[0] = -model.baseline();
    stereo_parameters.right_to_left_translation[1] = 0.0;
    stereo_parameters.right_to_left_translation[2] = 0.0;

    fovis::StereoCalibration* stereo_calibration = 
      new fovis::StereoCalibration(stereo_parameters);
    fovis::Rectification* rectification = 
      new fovis::Rectification(left_parameters);

    stereo_depth_.reset(
        new fovis::StereoDepth(stereo_calibration, visual_odometer_options_));
    visual_odometer_.reset(
        new fovis::VisualOdometry(rectification, visual_odometer_options_));

    getBaseToSensorTransform(l_info_msg->header.stamp, 
        l_info_msg->header.frame_id,
        initial_base_to_sensor_);

    ROS_INFO_STREAM("Initialized fovis stereo odometry.");
  }

  void getBaseToSensorTransform(const ros::Time& stamp, 
      const std::string& sensor_frame_id, tf::StampedTransform& base_to_sensor)
  {
    std::string error_msg;
    if (tf_listener_.canTransform(
          base_link_frame_id_, sensor_frame_id, stamp, &error_msg))
    {
      tf_listener_.lookupTransform(
          base_link_frame_id_,
          sensor_frame_id,
          stamp, base_to_sensor);
    }
    else
    {
      ROS_WARN_THROTTLE(10.0, "The tf from '%s' to '%s' does not seem to be "
                              "available, will assume it as identity!", 
                              base_link_frame_id_.c_str(),
                              sensor_frame_id.c_str());
      ROS_DEBUG("Transform error: %s", error_msg.c_str());
      base_to_sensor.setIdentity();
    }
  }

  void eigenToTF(const Eigen::Isometry3d& pose, tf::Transform& transform)
  {
    tf::Vector3 origin(
        pose.translation().x(), pose.translation().y(), pose.translation().z());
    Eigen::Quaterniond rotation(pose.rotation());
    tf::Quaternion quat(rotation.x(), rotation.y(),
        rotation.z(), rotation.w());
    transform = tf::Transform(quat, origin);
  }

  void imageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg)
  {
    bool first_run = false;
    // create odometer if not exists
    if (!visual_odometer_)
    {
      initOdometer(l_info_msg, r_info_msg);
      first_run = true;
    }

    // convert images if necessary
    uint8_t *l_image_data, *r_image_data;
    int l_step, r_step;
    cv_bridge::CvImageConstPtr l_cv_ptr, r_cv_ptr;
    l_cv_ptr = cv_bridge::toCvShare(l_image_msg, sensor_msgs::image_encodings::MONO8);
    l_image_data = l_cv_ptr->image.data;
    l_step = l_cv_ptr->image.step[0];
    r_cv_ptr = cv_bridge::toCvShare(r_image_msg, sensor_msgs::image_encodings::MONO8);
    r_image_data = r_cv_ptr->image.data;
    r_step = r_cv_ptr->image.step[0];

    ROS_ASSERT(l_step == r_step);
    ROS_ASSERT(l_step == static_cast<int>(l_image_msg->width));
    ROS_ASSERT(l_image_msg->width == r_image_msg->width);
    ROS_ASSERT(l_image_msg->height == r_image_msg->height);

    // pass images to odometer
    stereo_depth_->setRightImage(r_image_data);
    visual_odometer_->processFrame(l_image_data, stereo_depth_.get());

    // skip visualization on first run as no reference image is present
    if (!first_run && features_pub_.getNumSubscribers() > 0)
    {
      cv_bridge::CvImage cv_image;
      cv_image.header.stamp = l_image_msg->header.stamp;
      cv_image.header.frame_id = l_image_msg->header.frame_id;
      cv_image.encoding = sensor_msgs::image_encodings::BGR8;
      cv_image.image = visualization::paint(visual_odometer_.get());
      features_pub_.publish(cv_image.toImageMsg());
    }

    {
      // create and publish fovis info msg
      FovisInfo info_msg;
      info_msg.header.stamp = l_image_msg->header.stamp;
      info_msg.change_reference_frame = 
        visual_odometer_->getChangeReferenceFrames();
      info_msg.fast_threshold =
        visual_odometer_->getFastThreshold();
     const fovis::OdometryFrame* frame = 
        visual_odometer_->getTargetFrame();
      info_msg.num_total_detected_keypoints =
       frame->getNumDetectedKeypoints();
      info_msg.num_total_keypoints = frame->getNumKeypoints();
      info_msg.num_detected_keypoints.resize(frame->getNumLevels());
      info_msg.num_keypoints.resize(frame->getNumLevels());
      for (int i = 0; i < frame->getNumLevels(); ++i)
      {
        info_msg.num_detected_keypoints[i] =
          frame->getLevel(i)->getNumDetectedKeypoints();
        info_msg.num_keypoints[i] =
          frame->getLevel(i)->getNumKeypoints();
      }
      const fovis::MotionEstimator* estimator = 
        visual_odometer_->getMotionEstimator();
      info_msg.motion_estimate_status_code =
        estimator->getMotionEstimateStatus();
      info_msg.motion_estimate_status = 
        fovis::MotionEstimateStatusCodeStrings[
          info_msg.motion_estimate_status_code];
      info_msg.num_matches = estimator->getNumMatches();
      info_msg.num_inliers = estimator->getNumInliers();
      info_msg.num_reprojection_failures =
        estimator->getNumReprojectionFailures();
      info_msg.motion_estimate_valid = 
        estimator->isMotionEstimateValid();
      info_pub_.publish(info_msg);
    }

    // create odometry and pose messages
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = l_image_msg->header.stamp;
    odom_msg.header.frame_id = odom_frame_id_;
    odom_msg.child_frame_id = base_link_frame_id_;
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = l_image_msg->header.stamp;
    pose_msg.header.frame_id = base_link_frame_id_;

    // on success, start fill message and tf
    fovis::MotionEstimateStatusCode status = 
      visual_odometer_->getMotionEstimateStatus();
    if (status == fovis::SUCCESS)
    {
      // get pose and motion from odometer
      const Eigen::Isometry3d& pose = visual_odometer_->getPose();
      tf::Transform sensor_pose;
      eigenToTF(pose, sensor_pose);
      // calculate transform of odom to base based on base to sensor 
      // and sensor to sensor
      tf::StampedTransform current_base_to_sensor;
      getBaseToSensorTransform(
          l_image_msg->header.stamp, l_image_msg->header.frame_id, 
          current_base_to_sensor);
      tf::Transform base_transform = 
        initial_base_to_sensor_ * sensor_pose * current_base_to_sensor.inverse();

      // publish transform
      if (publish_tf_)
      {
        tf_broadcaster_.sendTransform(
            tf::StampedTransform(base_transform, l_image_msg->header.stamp,
            odom_frame_id_, base_link_frame_id_));
      }

      // fill odometry and pose msg
      tf::poseTFToMsg(base_transform, odom_msg.pose.pose);
      pose_msg.pose = odom_msg.pose.pose;

      // can we calculate velocities?
      double dt = last_time_.isZero() ? 
        0.0 : (l_image_msg->header.stamp - last_time_).toSec();
      if (dt > 0.0)
      {
        const Eigen::Isometry3d& motion = visual_odometer_->getMotionEstimate();
        tf::Transform sensor_motion;
        eigenToTF(motion, sensor_motion);
        // in theory the first factor would have to be base_to_sensor of t-1
        // and not of t (irrelevant for static base to sensor anyways)
        tf::Transform delta_base_transform = 
          current_base_to_sensor * sensor_motion * current_base_to_sensor.inverse();
        // calculate twist from delta transform
        odom_msg.twist.twist.linear.x = delta_base_transform.getOrigin().getX() / dt;
        odom_msg.twist.twist.linear.y = delta_base_transform.getOrigin().getY() / dt;
        odom_msg.twist.twist.linear.z = delta_base_transform.getOrigin().getZ() / dt;
        tf::Quaternion delta_rot = delta_base_transform.getRotation();
        double angle = delta_rot.getAngle();
        tf::Vector3 axis = delta_rot.getAxis();
        tf::Vector3 angular_twist = axis * angle / dt;
        odom_msg.twist.twist.angular.x = angular_twist.x();
        odom_msg.twist.twist.angular.y = angular_twist.y();
        odom_msg.twist.twist.angular.z = angular_twist.z();

        // add covariance
        const Eigen::MatrixXd& motion_cov = visual_odometer_->getMotionEstimateCov();
        for (int i=0;i<6;i++)
          for (int j=0;j<6;j++)
            odom_msg.twist.covariance[j*6+i] = motion_cov(i,j);
      }
      // TODO integrate covariance for pose covariance
      last_time_ = l_image_msg->header.stamp;
    }
    else
    {
      ROS_ERROR_STREAM("fovis stereo odometry failed: " << 
          fovis::MotionEstimateStatusCodeStrings[status]);
      last_time_ = ros::Time(0);
    }
    odom_pub_.publish(odom_msg);
    pose_pub_.publish(pose_msg);
  }
};

} // end of namespace


int main(int argc, char **argv)
{
  ros::init(argc, argv, "stereo_odometer");
  if (ros::names::remap("stereo") == "stereo") {
    ROS_WARN("'stereo' has not been remapped! Example command-line usage:\n"
             "\t$ rosrun fovis_ros stereo_odometer stereo:=narrow_stereo image:=image_rect");
  }
  if (ros::names::remap("image").find("rect") == std::string::npos) {
    ROS_WARN("stereo_odometer needs rectified input images. The used image "
             "topic is '%s'. Are you sure the images are rectified?",
             ros::names::remap("image").c_str());
  }

  std::string transport = argc > 1 ? argv[1] : "raw";
  fovis_ros::StereoOdometer odometer(transport);
  
  ros::spin();
  return 0;
}

