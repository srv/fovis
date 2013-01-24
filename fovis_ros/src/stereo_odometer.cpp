#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/stereo_camera_model.h>
#include <cv_bridge/cv_bridge.h>

#include <fovis/visual_odometry.hpp>
#include <fovis/stereo_depth.hpp>

#include "stereo_processor.h"

// to remove after debugging
#include <opencv2/highgui/highgui.hpp>

namespace fovis_ros
{

// some arbitrary values (0.1m linear cov. 10deg. angular cov.)
static const boost::array<double, 36> STANDARD_POSE_COVARIANCE =
{ { 0.1, 0, 0, 0, 0, 0,
    0, 0.1, 0, 0, 0, 0,
    0, 0, 0.1, 0, 0, 0,
    0, 0, 0, 0.17, 0, 0,
    0, 0, 0, 0, 0.17, 0,
    0, 0, 0, 0, 0, 0.17 } };
static const boost::array<double, 36> STANDARD_TWIST_COVARIANCE =
{ { 0.05, 0, 0, 0, 0, 0,
    0, 0.05, 0, 0, 0, 0,
    0, 0, 0.05, 0, 0, 0,
    0, 0, 0, 0.09, 0, 0,
    0, 0, 0, 0, 0.09, 0,
    0, 0, 0, 0, 0, 0.09 } };
static const boost::array<double, 36> BAD_COVARIANCE =
{ { 9999, 0, 0, 0, 0, 0,
    0, 9999, 0, 0, 0, 0,
    0, 0, 9999, 0, 0, 0,
    0, 0, 0, 9999, 0, 0,
    0, 0, 0, 0, 9999, 0,
    0, 0, 0, 0, 0, 9999 } };


class StereoOdometer : public StereoProcessor
{

private:

  boost::shared_ptr<fovis::VisualOdometry> visual_odometer_;
  fovis::VisualOdometryOptions visual_odometer_options_;
  boost::shared_ptr<fovis::StereoDepth> stereo_depth_;

public:

  StereoOdometer(const std::string& transport) : 
    StereoProcessor(transport), 
    visual_odometer_options_(fovis::VisualOdometry::getDefaultOptions())
  {
    // TODO load parameters from node handle to visual_odometer_options_
    ros::NodeHandle local_nh("~");

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
 
    /*
    bool first_run = false;
    // create odometer if not exists
    if (!visual_odometer_)
    {
      first_run = true;
      initOdometer(l_info_msg, r_info_msg);
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
    ROS_ASSERT(l_image_msg->width == r_image_msg->width);
    ROS_ASSERT(l_image_msg->height == r_image_msg->height);

    int32_t dims[] = {l_image_msg->width, l_image_msg->height, l_step};
    // on first run or when odometer got lost, only feed the odometer with 
    // images without retrieving data
    if (first_run || got_lost_)
    {
      visual_odometer_->process(l_image_data, r_image_data, dims);
      got_lost_ = false;
      // on first run publish zero once
      if (first_run)
      {
        tf::Transform delta_transform;
        delta_transform.setIdentity();
        integrateAndPublish(delta_transform, l_image_msg->header.stamp);
      }
    }
    else
    {
      bool success = visual_odometer_->process(
          l_image_data, r_image_data, dims, last_motion_small_);
      if (success)
      {
        Matrix motion = Matrix::inv(visual_odometer_->getMotion());
        ROS_DEBUG("Found %i matches with %i inliers.", 
                  visual_odometer_->getNumberOfMatches(),
                  visual_odometer_->getNumberOfInliers());
        ROS_DEBUG_STREAM("fovis returned the following motion:\n" << motion);
        Matrix camera_motion;
        // if image was replaced due to small motion we have to subtract the 
        // last motion to get the increment
        if (last_motion_small_)
        {
          camera_motion = Matrix::inv(reference_motion_) * motion;
        }
        else
        {
          // image was not replaced, report full motion from odometer
          camera_motion = motion;
        }
        reference_motion_ = motion; // store last motion as reference

        // calculate current feature flow
        std::vector<Matcher::p_match> matches = visual_odometer_->getMatches();
        std::vector<int> inlier_indices = visual_odometer_->getInlierIndices();
        double feature_flow = computeFeatureFlow(matches);
        last_motion_small_ = (feature_flow < motion_threshold_);
        ROS_DEBUG_STREAM("Feature flow is " << feature_flow 
            << ", marking last motion as " 
            << (last_motion_small_ ? "small." : "normal."));

        btMatrix3x3 rot_mat(
          camera_motion.val[0][0], camera_motion.val[0][1], camera_motion.val[0][2],
          camera_motion.val[1][0], camera_motion.val[1][1], camera_motion.val[1][2],
          camera_motion.val[2][0], camera_motion.val[2][1], camera_motion.val[2][2]);
        btVector3 t(camera_motion.val[0][3], camera_motion.val[1][3], camera_motion.val[2][3]);
        tf::Transform delta_transform(rot_mat, t);

        setPoseCovariance(STANDARD_POSE_COVARIANCE);
        setTwistCovariance(STANDARD_TWIST_COVARIANCE);

        integrateAndPublish(delta_transform, l_image_msg->header.stamp);

        if (point_cloud_pub_.getNumSubscribers() > 0)
        {
          computeAndPublishPointCloud(l_info_msg, l_image_msg, r_info_msg, matches, inlier_indices);
        }
      }
      else
      {
        setPoseCovariance(BAD_COVARIANCE);
        setTwistCovariance(BAD_COVARIANCE);
        tf::Transform delta_transform;
        delta_transform.setIdentity();
        integrateAndPublish(delta_transform, l_image_msg->header.stamp);

        ROS_DEBUG("Call to VisualOdometryStereo::process() failed.");
        ROS_WARN_THROTTLE(1.0, "Visual Odometer got lost!");
        got_lost_ = true;
      }
    }
  */
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

