#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <fovis/visual_odometry.hpp>
#include <fovis/stereo_depth.hpp>

#include "stereo_processor.h"

namespace fovis_ros
{

class StereoOdometer : public StereoProcessor
{

private:

  boost::shared_ptr<fovis::VisualOdometry> visual_odometer_;
  fovis::VisualOdometryOptions visual_odometer_options_;
  boost::shared_ptr<fovis::StereoDepth> stereo_depth_;

  ros::Time last_time_;

  // publisher
  ros::Publisher odom_pub_;
  ros::Publisher pose_pub_;

public:

  StereoOdometer(const std::string& transport) : 
    StereoProcessor(transport), 
    visual_odometer_options_(fovis::VisualOdometry::getDefaultOptions())
  {
    // TODO load parameters from node handle to visual_odometer_options_
    ros::NodeHandle local_nh("~");

    odom_pub_ = local_nh.advertise<nav_msgs::Odometry>("odometry", 1);
    pose_pub_ = local_nh.advertise<geometry_msgs::PoseStamped>("pose", 1);
  }

protected:

  void initOdometer(
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg)
  {
    // read calibration info from camera info message
    // to fill remaining parameters
    image_geometry::StereoCameraModel model;
    model.fromCameraInfo(*l_info_msg, *r_info_msg);
    // TODO initialize stereo_depth_ somewhat like this:
    /*
    visual_odometer_params_.base = model.baseline();
    visual_odometer_params_.calib.f = model.left().fx();
    visual_odometer_params_.calib.cu = model.left().cx();
    visual_odometer_params_.calib.cv = model.left().cy();
    visual_odometer_.reset(new VisualOdometryStereo(visual_odometer_params_));
    if (l_info_msg->header.frame_id != "") setSensorFrameId(l_info_msg->header.frame_id);
    ROS_INFO_STREAM("Initialized fovis stereo odometry "
                    "with the following parameters:" << std::endl << 
                    visual_odometer_params_ << 
                    "  motion_threshold = " << motion_threshold_);
                    */
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

    fovis::MotionEstimateStatusCode status = 
      visual_odometer_->getMotionEstimateStatus();

    if (status == fovis::SUCCESS)
    {
      // get pose and put it into messages
      const Eigen::Isometry3d& pose = visual_odometer_->getPose();
      Eigen::Vector3d translation = pose.translation();
      Eigen::Quaterniond rotation(pose.rotation());

      nav_msgs::Odometry odom_msg;
      odom_msg.header.stamp = l_image_msg->header.stamp;
      odom_msg.header.frame_id = "/odom";
      odom_msg.child_frame_id = l_image_msg->header.frame_id;
      odom_msg.pose.pose.position.x = translation.x();
      odom_msg.pose.pose.position.y = translation.y();
      odom_msg.pose.pose.position.z = translation.z();
      odom_msg.pose.pose.orientation.x = rotation.x();
      odom_msg.pose.pose.orientation.y = rotation.y();
      odom_msg.pose.pose.orientation.z = rotation.z();
      odom_msg.pose.pose.orientation.w = rotation.w();

      if (!last_time_.isZero())
      {
        float dt = (l_image_msg->header.stamp - last_time_).toSec();
        if (dt < 0.0)
        {
          const Eigen::Isometry3d& last_motion = 
            visual_odometer_->getMotionEstimate();
          Eigen::Vector3d last_translation = last_motion.translation();
          odom_msg.twist.twist.linear.x = last_translation.x() / dt;
          odom_msg.twist.twist.linear.y = last_translation.y() / dt;
          odom_msg.twist.twist.linear.z = last_translation.z() / dt;
          // TODO insert angular twist calculation
        }
      }

      const Eigen::MatrixXd& motion_cov = 
        visual_odometer_->getMotionEstimateCov();
      for (int i=0;i<6;i++)
        for (int j=0;j<6;j++)
          odom_msg.twist.covariance[j*6+i] = motion_cov(i,j);
      // TODO integrate covariance for pose covariance
      
      odom_pub_.publish(odom_msg);
      geometry_msgs::PoseStamped pose_msg;
      pose_msg.pose = odom_msg.pose.pose;
      pose_msg.header = odom_msg.header;
      pose_msg.header.frame_id = odom_msg.child_frame_id;
      pose_pub_.publish(pose_msg);
      
      last_time_ = l_image_msg->header.stamp;
    }
    else
    {
      ROS_ERROR_STREAM("fovis stereo odometry failed: " << 
          fovis::MotionEstimateStatusCodeStrings[status]);
    }
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

