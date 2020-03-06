#ifndef AUV_2_ROS_HPP
#define AUV_2_ROS_HPP 

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cereal/archives/binary.hpp>

#include "data_tools/std_data.h"

#include "submaps_tools/cxxopts.hpp"
#include "submaps_tools/submaps.hpp"
#include "registration/utils_visualization.hpp"
#include "meas_models/multibeam_simple.hpp"

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#include <actionlib/client/simple_action_client.h>
#include <auv_2_ros/MbesSimAction.h>


using namespace Eigen;
using namespace std;

class BathymapConstructor{

public:
    BathymapConstructor(std::string node_name, ros::NodeHandle &nh);
    ~BathymapConstructor();

    void init(const boost::filesystem::path auv_path);

    void run();

    void broadcastTf(const ros::TimerEvent &);

    void publishOdom(Vector3d odom_ping_i, Vector3d euler);

private:
    std::string node_name_;
    ros::NodeHandle* nh_;

    ros::Publisher ping_pub_;
    ros::Publisher sim_ping_pub_;
    ros::Publisher map_pub_;
    ros::Publisher test_pub_;
    ros::Publisher odom_pub_;

    actionlib::SimpleActionClient<auv_2_ros::MbesSimAction>* ac_;

    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener* tfListener_;
    tf::TransformListener tflistener_;
    tf2_ros::StaticTransformBroadcaster static_broadcaster_;
    tf2_ros::TransformBroadcaster br_;
    std::vector<geometry_msgs::TransformStamped> pings_tf_;

    Eigen::Isometry3d map_tf_;
    Eigen::Isometry3d odom_tf_;

    SubmapsVec maps_gt_;
    SubmapsVec traj_pings_;

    MultibeamSensor<PointT> vox_oc_;

    ros::Time time_now_, time_prev_;
    geometry_msgs::TransformStamped prev_base_link_;
    geometry_msgs::TransformStamped new_base_link_;

    std::string world_frame_, map_frame_, odom_frame_, base_frame_, mbes_frame_;

    int ping_num_;

};


#endif