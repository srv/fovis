cmake_minimum_required(VERSION 2.8.3)
project(fovis_ros)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

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

find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})

find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS signals thread)

add_message_files(DIRECTORY msg FILES FovisInfo.msg)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS message_runtime)

IF(SSSE3_FOUND)
	add_definitions("-msse3")
ENDIF(SSSE3_FOUND)

include_directories(src
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS} 
	${PCL_ICNLUDE_DIRS})

add_library(visualization src/visualization.cpp)

add_executable(fovis_stereo_odometer src/stereo_odometer.cpp)

add_executable(fovis_mono_depth_odometer src/mono_depth_odometer.cpp)

add_dependencies(fovis_mono_depth_odometer ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(fovis_stereo_odometer ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(visualization ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(visualization ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(fovis_stereo_odometer ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} visualization)
target_link_libraries(fovis_mono_depth_odometer ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} visualization)

