cmake_minimum_required(VERSION 2.8.3)
project(fovis_ros)

find_package(catkin REQUIRED COMPONENTS 
	libfovis
  roscpp 
  sensor_msgs 
  nav_msgs 
  message_filters 
  image_transport 
  cv_bridge 
  image_geometry 
  tf 
  pcl_ros 
  std_srvs
  message_generation
  std_msgs)
  #genmsg)

find_package(OpenCV REQUIRED)
find_package(PCL 1.7.0 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS signals thread)
include_directories(${Boost_INCLUDE_DIRS})

add_message_files(DIRECTORY msg FILES FovisInfo.msg)

generate_messages(DEPENDENCIES std_msgs)

catkin_package()

add_definitions("-msse3")

include_directories(src ${libfovis_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

add_library(visualization src/visualization.cpp)

add_executable(fovis_stereo_odometer src/stereo_odometer.cpp)

add_executable(fovis_mono_depth_odometer src/mono_depth_odometer.cpp)

target_link_libraries(fovis_stereo_odometer ${catkin_LIBRARIES} ${PCL_LIBRARIES} visualization)
target_link_libraries(fovis_mono_depth_odometer ${catkin_LIBRARIES} ${PCL_LIBRARIES} visualization)

