#include <ros/ros.h>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

//#include <pcl/ros/conversions.h>
//#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <pcl/filters/extract_indices.h>

#include "rosbag/bag.h"
#include "rosbag/query.h"
#include "rosbag/view.h"

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include <Eigen/Dense>

using namespace std;

tf::TransformListener *listener_ = 0;
ros::Publisher pct_pub;
ros::Publisher parr_pub;
ros::Publisher parr_flat_pub;
ros::Publisher marker_pub;
ros::Publisher marker_pub_arr;

void getCloud(sensor_msgs::PointCloud2 &cloud_msg, std::string frame_id, ros::Time after, ros::Time *tm)
{

    sensor_msgs::PointCloud2 pc;// = *(ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/kinect/cloud_throttled"));
    bool found = false;
    while (!found)
    {
        //pc  = *(ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/kinect/cloud_throttled"));
        ROS_INFO("BEFORE");
        pc  = *(ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/kinect/cloud_throttled"));
        ROS_INFO("AFTER");
        if ((after == ros::Time(0,0)) || (pc.header.stamp > after))
            found = true;
        else
        {
            ROS_ERROR("getKinectCloudXYZ cloud too old : stamp %f , target time %f",pc.header.stamp.toSec(), after.toSec());
        }
    }
    if (tm)
        *tm = pc.header.stamp;

    //sensor_msgs::PointCloud2 pct; //in map frame

    pcl_ros::transformPointCloud(frame_id,pc,cloud_msg,*listener_);
    cloud_msg.header.frame_id = frame_id.c_str();

    //rosbag::Bag bag;
    //bag.open("single_cloud.bag", rosbag::bagmode::Write);
    //bag.write("cloud", ros::Time::now(), pct);
    //bag.close();
    //pcl::fromROSMsg(pct, *cloud);
    ROS_INFO("got a cloud");
}

int marker_ids_ = 0;

// add a "gripper" marker
void addMarker(visualization_msgs::MarkerArray &marr, tf::Pose pose, double x = 0.05, double y= 0.05, double z= 0.05)
{
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time();
    marker.ns = "nop";
    marker.id = marker_ids_++;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = pose.getOrigin().x();
    marker.pose.position.y = pose.getOrigin().y();
    marker.pose.position.z = pose.getOrigin().z();
    marker.pose.orientation.x = pose.getRotation().x();
    marker.pose.orientation.y = pose.getRotation().y();
    marker.pose.orientation.z = pose.getRotation().z();
    marker.pose.orientation.w = pose.getRotation().w();
    marker.scale.x = 0.01;
    marker.scale.y = 0.01;
    marker.scale.z = z;
    marker.color.a = 1;
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
//only if using a MESH_RESOURCE marker type:
    marr.markers.push_back(marker);

    marker.id = marker_ids_++;
    marker.scale.x = x;
    marker.scale.y = 0.01;
    marker.scale.z = 0.01;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    marr.markers.push_back(marker);

    marker.id = marker_ids_++;
    marker.scale.x = 0.01;
    marker.scale.y = y;
    marker.scale.z = 0.01;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marr.markers.push_back(marker);

    tf::Pose rel;

    rel.setRotation(tf::Quaternion(0,0,0,1));

    marker.scale.x = 0.05;
    marker.scale.y = 0.01;
    marker.scale.z = 0.01;
    marker.color.r = 0.90;
    marker.color.g = 0.50;
    marker.color.b = 0.90;
    marker.color.a = 1;

    marker.type = visualization_msgs::Marker::CUBE;

    rel.setOrigin(tf::Vector3(-0.025,0.03,0));
    rel = pose * rel;
    marker.pose.position.x = rel.getOrigin().x();
    marker.pose.position.y = rel.getOrigin().y();
    marker.pose.position.z = rel.getOrigin().z();
    marker.id = marker_ids_++;
    marr.markers.push_back(marker);

    rel.setOrigin(tf::Vector3(-0.025,-0.03,0));
    rel = pose * rel;
    marker.pose.position.x = rel.getOrigin().x();
    marker.pose.position.y = rel.getOrigin().y();
    marker.pose.position.z = rel.getOrigin().z();
    marker.id = marker_ids_++;
    marr.markers.push_back(marker);

    rel.setOrigin(tf::Vector3(-0.05,0,0));
    rel = pose * rel;
    marker.scale.x = 0.01;
    marker.scale.y = 0.06;
    marker.pose.position.x = rel.getOrigin().x();
    marker.pose.position.y = rel.getOrigin().y();
    marker.pose.position.z = rel.getOrigin().z();
    marker.id = marker_ids_++;
    marr.markers.push_back(marker);




}

using namespace Eigen;

template <typename Derived, typename OtherDerived>
void calculateSampleCovariance(const MatrixBase<Derived>& x, const MatrixBase<Derived>& y, MatrixBase<OtherDerived> & C_)
{
    typedef typename Derived::Scalar Scalar;
    typedef typename internal::plain_row_type<Derived>::type RowVectorType;

    const Scalar num_observations = static_cast<Scalar>(x.rows());

    const RowVectorType x_mean = x.colwise().sum() / num_observations;
    const RowVectorType y_mean = y.colwise().sum() / num_observations;

    MatrixBase<OtherDerived>& C = const_cast< MatrixBase<OtherDerived>& >(C_);

    C.derived().resize(x.cols(),x.cols()); // resize the derived object
    C = (x.rowwise() - x_mean).transpose() * (y.rowwise() - y_mean) / num_observations;


}

MatrixXd pos_covar_xy(std::vector<tf::Vector3> points)
{
    MatrixXd esamples(points.size(),3);
    for (size_t n=0; n < points.size(); ++n)
    {
        VectorXd evec;
        evec.resize(3);
        evec(0) = points[n].x();
        evec(1) = points[n].y();
        evec(2) = 0;// points[n].z();
        esamples.row(n) = evec;
    }
    MatrixXd ret;
    calculateSampleCovariance(esamples,esamples,ret);
    return ret;
}

void pos_eigen_xy(std::vector<tf::Vector3> points, std::vector<tf::Vector3> &evec, std::vector<double> &eval)
{
    Matrix<double,3,3> covarMat = pos_covar_xy(points);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,3,3> >
    eigenSolver(covarMat);
    //cout << "vec" << endl << eigenSolver.eigenvectors() << endl;
    //cout << "val" << endl << eigenSolver.eigenvalues() << endl;
    MatrixXf::Index  maxIndex;
    eigenSolver.eigenvalues().maxCoeff(&maxIndex);
    MatrixXd max = eigenSolver.eigenvectors().col(maxIndex);
    //cout << "max " << endl << max << endl;
    tf::Vector3 maxVec(max(0),max(1),max(2));
    evec.push_back(maxVec);
}


// z plus red/blue axis
tf::Vector3 pcl_to_zcol(pcl::PointXYZRGB pt)
{
    return tf::Vector3(pt.z, (pt.b - pt.r) / 1500.0f, 0);
    //return tf::Vector3(pt.z, 0, 0);
}

void pos_eigen_zcol(std::vector<int> &idx, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<tf::Vector3> &evec, std::vector<double> &eval, bool print = false)
{
    std::vector<tf::Vector3> projected; // points projected to z+col space
    for (std::vector<int>::iterator it = idx.begin(); it != idx.end(); it ++)
        projected.push_back(pcl_to_zcol(cloud->points[*it]));

    Matrix<double,3,3> covarMat = pos_covar_xy(projected);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,3,3> >
    eigenSolver(covarMat);
    MatrixXf::Index  maxIndex;
    eigenSolver.eigenvalues().maxCoeff(&maxIndex);
    MatrixXd max = eigenSolver.eigenvectors().col(maxIndex);
    if (print) {
      cout << "vec" << endl << eigenSolver.eigenvectors() << endl;
      cout << "val" << endl << eigenSolver.eigenvalues() << endl;
      cout << "max " << endl << max << endl;
    }
    tf::Vector3 maxVec(max(0),max(1),max(2));
    evec.push_back(maxVec);
}


/*void getCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string frame_id, ros::Time after, ros::Time *tm)
{

    sensor_msgs::PointCloud2 pc;// = *(ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/kinect/cloud_throttled"));
    bool found = false;
    while (!found)
    {
        //pc  = *(ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/kinect/cloud_throttled"));
        ROS_INFO("BEFORE");
        pc  = *(ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/kinect/cloud_throttled"));
        ROS_INFO("AFTER");
        if ((after == ros::Time(0,0)) || (pc.header.stamp > after))
            found = true;
        else
        {
            ROS_ERROR("getKinectCloudXYZ cloud too old : stamp %f , target time %f",pc.header.stamp.toSec(), after.toSec());
        }
    }
    if (tm)
        *tm = pc.header.stamp;

    sensor_msgs::PointCloud2 pct; //in map frame

    pcl_ros::transformPointCloud(frame_id,pc,pct,*listener_);
    pct.header.frame_id = frame_id.c_str();

    rosbag::Bag bag;
    bag.open("single_cloud.bag", rosbag::bagmode::Write);
    bag.write("cloud", ros::Time::now(), pct);
    bag.close();

    pcl::fromROSMsg(pct, *cloud);
    ROS_INFO("got a cloud");
}*/


//! get top grasp points on rims of objects
void classify_cloud(sensor_msgs::PointCloud2 msg)
{
    ROS_INFO("classify");
    float field[100 * 100]; // we start with 1x1m 1cm resolution
    std::vector<tf::Vector3> field_topvec[100 * 100];
    std::vector<tf::Vector3> field_botvec[100 * 100];

    // field of maximum z values in an x/y bin
    for (size_t i = 0; i < 100*100; ++i)
    {
        field[i] = 0;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr z_maxima(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(msg, *cloud);

    //! get maxima in z by iterating over all points and updating the bin the points fall into
    //! we are looking at a 1mx1m area of hopefully table, the scaling defines the resolution
    //! and the shift defines where we are looking at, so far its always aligned with the msg
    //! coordinate frame, so moving around could also be done by transforming the msg to
    //! another coord frame.
    double scaling = 20; // 1 cm resolution => scaling 100  10cm => scaling 10
    tf::Vector3 shift(2.5,-1,0); // shift the data
    for (size_t i=0; i < cloud->points.size(); ++i)
    {
        tf::Vector3 act(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z);
        act = act + shift;
        if ((act.x() > 0) && (act.y() > 0) && (act.y() < 1) && (act.x() < 1))
        {
            int xcoord = act.x() * scaling;
            int ycoord = act.y() * scaling;
            if (field[xcoord + ycoord * 100] < act.z())
                field[xcoord + ycoord * 100] = act.z();
        }
    }

    // index set of points that are interesting for us later => for debugging
    boost::shared_ptr<std::vector<int> > indices( new std::vector<int> );

    //! we look at points that lie within delta from the top point in the bin
    //! this set is divided into points below 1/3 of delta and above 2/3 of delta
    //! forming the top and bottom sets
    double delta = .04;

    for (size_t i=0; i < cloud->points.size(); ++i)
    {
        tf::Vector3 act_(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z);
        tf::Vector3 act = act_ + shift;
        if ((act.x() > 0) && (act.y() > 0) && (act.y() < 1) && (act.x() < 1))
        {
            int xcoord = act.x() * scaling;
            int ycoord = act.y() * scaling;
            size_t addr = xcoord + ycoord * 100;
            //if (fabs(field[xcoord + ycoord * 100] - act.z()) < 0.0000000001)
            double dist_to_max = fabs(field[xcoord + ycoord * 100] - act.z());
            float fac =  dist_to_max / .02;
            if (dist_to_max < delta / 3)
            {
                indices->push_back(i);
                cloud->points[i].r =  0;
                cloud->points[i].g =  0;
                cloud->points[i].b =  255;
                field_topvec[addr].push_back(act_);
                //field_topvec[i] += act_;
                //field_topvec_n[i] += 1;
            }

            if ((dist_to_max > 2 * delta / 3) && (dist_to_max < delta))
            {
                indices->push_back(i);
                cloud->points[i].r =  255;
                cloud->points[i].g =  0;
                cloud->points[i].b =  0;
                field_botvec[addr].push_back(act_);
                //field_botvec[i] += act_;
                //field_botvec_n[i] += 1;
            }
        }
    }

    geometry_msgs::PoseArray parr;
    parr.header.frame_id = msg.header.frame_id;
    parr.header.stamp = ros::Time(0);

    visualization_msgs::MarkerArray marr;


    for (size_t y = 0; y < scaling; ++y)
    {
        for (size_t x = 0; x < scaling; ++x)
        {
            size_t addr = x + y * 100;
            // for each bin, check if we have some points int the top and bottom set
            if ((field_topvec[addr].size() > 10) && (field_botvec[addr].size() > 10))
            {
                tf::Vector3 top(0,0,0);
                tf::Vector3 bot(0,0,0);
                //! calculate means of the top and bottom points and take them to set the up direction / slope of the bowls etc.
                for (std::vector<tf::Vector3>::iterator it = field_topvec[addr].begin(); it != field_topvec[addr].end(); ++it)
                    top += *it;
                top /= field_topvec[addr].size();
                for (std::vector<tf::Vector3>::iterator it = field_botvec[addr].begin(); it != field_botvec[addr].end(); ++it)
                    bot += *it;
                bot /= field_botvec[addr].size();

                //!todo, this is depending on the table height!
                if (top.z() < 0.9)
                    continue;

                std::vector<tf::Vector3> evec;
                std::vector<double> eval;
                //! calculate a pca of the top point set and take that as the "right" axis of the grasp
                pos_eigen_xy(field_topvec[addr], evec, eval);

                tf::Vector3 z_axis = evec[0].normalize();
                tf::Vector3 x_axis = top - bot; // tf::Vector3(0,0,1);
                tf::Vector3 y_axis = (x_axis.cross(z_axis)).normalize();
                x_axis = (z_axis.cross(y_axis)).normalize();
                //z_axis = tf::Vector3(0,0,1);

                //ROS_INFO("/x_AXIS %f %f %f len %f", x_axis.x(), x_axis.y(), x_axis.z(), x_axis.length());
                //ROS_INFO("/z_AXIS %f %f %f len %f", z_axis.x(), z_axis.y(), z_axis.z(), z_axis.length());

                y_axis = (x_axis.cross(z_axis)).normalize();
                x_axis = (y_axis.cross(z_axis)).normalize();

                //ROS_INFO("x_AXIS %f %f %f len %f", x_axis.x(), x_axis.y(), x_axis.z(), x_axis.length());
                //ROS_INFO("y_AXIS %f %f %f len %f", y_axis.x(), y_axis.y(), y_axis.z(), y_axis.length());
                //ROS_INFO("z_AXIS %f %f %f len %f", z_axis.x(), z_axis.y(), z_axis.z(), z_axis.length());

                btMatrix3x3 rot(x_axis.x(), y_axis.x(), z_axis.x(),
                                x_axis.y(), y_axis.y(), z_axis.y(),
                                x_axis.z(), y_axis.z(), z_axis.z());
                btQuaternion rot_quat;
                rot.getRotation(rot_quat);

                tf::Pose pose;
                pose.setOrigin((bot + top) / 2);
                pose.setRotation(rot_quat);
                geometry_msgs::Pose pose_msg;
                tf::poseTFToMsg(pose,pose_msg);
                parr.poses.push_back(pose_msg);
                //std::cout << "bin/ias_drawer_executive -3 0 " << pose_msg.position.x << " " << pose_msg.position.y << " " << pose_msg.position.z + .1 << " "
                  //        << pose_msg.orientation.x << " " << pose_msg.orientation.y << " " << pose_msg.orientation.z << " " << pose_msg.orientation.w <<  std::endl;
                //std::cout << "bin/ias_drawer_executive -3 0 " << pose_msg.position.x << " " << pose_msg.position.y << " " << pose_msg.position.z << " "
                  //        << pose_msg.orientation.x << " " << pose_msg.orientation.y << " " << pose_msg.orientation.z << " " << pose_msg.orientation.w <<  std::endl << std::endl;

                addMarker(marr, pose);
                //pose.setOrigin(bot + tf::Vector3(0.005,0.005,0));
                //tf::poseTFToMsg(pose,pose_msg);
                //parr.poses.push_back(pose_msg);

            }
        }
    }


    marker_pub_arr.publish(marr);


    parr_pub.publish(parr);

    pcl::ExtractIndices<pcl::PointXYZRGB> ei;
    ei.setInputCloud(cloud);
    ei.setIndices(indices);
    ei.filter(*z_maxima);

    sensor_msgs::PointCloud2 z_max_msg;
    pcl::toROSMsg(*z_maxima, z_max_msg);
    z_max_msg.header = msg.header;
    //pct_pub.publish(z_max_msg);

}



// get the low objects that aren't much higher than the table
void classify_cloud_low(sensor_msgs::PointCloud2 msg)
{
    ROS_INFO("classify_low");
    float field[100 * 100]; // we start with 1x1m 1cm resolution
    float field_low[100 * 100]; // we start with 1x1m 1cm resolution
    float field_avg[100 * 100]; // we start with 1x1m 1cm resolution
    float field_sigma[100 * 100]; // we start with 1x1m 1cm resolution
    float field_num[100 * 100]; // we start with 1x1m 1cm resolution
    std::vector<int> field_topvec[100 * 100];
    std::vector<int> field_botvec[100 * 100];
    std::vector<int> field_idx[100 * 100]; // idices on tops cloud showing which points lie in which bin

    tf::Vector3 shift(2.5,-1,0);

    for (size_t i = 0; i < 100*100; ++i)
    {
        field[i] = 0;
        field_low[i] = 1000;
        field_avg[i] = 0;
        field_num[i] = 0;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tops(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr z_maxima(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(msg, *cloud);

    //get maxima in z
    double scaling = 50; // 1 cm = 100

    //! get the top points' z coord per bin
    for (size_t i=0; i < cloud->points.size(); ++i)
    {
        tf::Vector3 act(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z);
        act = act + shift;
        if ((act.x() > 0) && (act.y() > 0) && (act.y() < 1) && (act.x() < 1))
        {
            int xcoord = act.x() * scaling;
            int ycoord = act.y() * scaling;
            size_t addr = xcoord + ycoord * 100;
            if (field[addr] < act.z())
                field[addr] = act.z();
            //if (field_low[addr] > act.z())
            //    field_low[addr] = act.z();
        }
    }

    boost::shared_ptr<std::vector<int> > top_indices( new std::vector<int> );

    double thickness = 0.04;

    //extract points within thickness cm of top point to get rid of thing underneath table etc
    for (size_t i=0; i < cloud->points.size(); ++i)
    {
        tf::Vector3 act(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z);
        act = act + shift;
        if ((act.x() > 0) && (act.y() > 0) && (act.y() < 1) && (act.x() < 1))
        {
            int xcoord = act.x() * scaling;
            int ycoord = act.y() * scaling;
            size_t addr = xcoord + ycoord * 100;
            //if (field[addr] - field_low[addr] < 0.1)
            //if (act.z() > 0.8)
            //if (act.z() < 0.88)
            if (act.z() - field[addr] > -thickness)
            {
                top_indices->push_back(i);
            }
        }
    }

    //! GET RID OF POINTS LOWER THAN thickness cms below top

    pcl::ExtractIndices<pcl::PointXYZRGB> ei2;
    ei2.setInputCloud(cloud);
    ei2.setIndices(top_indices);
    ei2.filter(*tops);

    for (size_t i = 0; i < 100*100; ++i)
    {
        field[i] = 0;
        field_low[i] = 1000;
        field_avg[i] = 0;
        field_num[i] = 0;
        field_sigma[i] = 0;
    }

    for (size_t i=0; i < tops->points.size(); ++i)
    {
        tf::Vector3 act(tops->points[i].x,tops->points[i].y,tops->points[i].z);
        act = act + shift;
        if ((act.x() > 0) && (act.y() > 0) && (act.y() < 1) && (act.x() < 1))
        {
            int xcoord = act.x() * scaling;
            int ycoord = act.y() * scaling;
            size_t addr = xcoord + ycoord * 100;
            if (field[addr] < act.z())
                field[addr] = act.z();
            if (field_low[addr] > act.z())
                field_low[addr] = act.z();
        }
    }

    for (size_t i=0; i < tops->points.size(); ++i)
    {
        tf::Vector3 act(tops->points[i].x,tops->points[i].y,tops->points[i].z);
        act = act + shift;
        if ((act.x() > 0) && (act.y() > 0) && (act.y() < 1) && (act.x() < 1))
        {
            int xcoord = act.x() * scaling;
            int ycoord = act.y() * scaling;
            size_t addr = xcoord + ycoord * 100;
            //if (field[addr] - field_low[addr] < 0.0)
            //  if (act.z() < 0.88)
            //    if (act.z() - field[addr] > -.02)
            // {
            field_avg[addr] += act.z();
            field_num[addr] ++;
            //  }
        }
    }


    for (size_t y = 0; y < scaling; ++y)
    {
        for (size_t x = 0; x < scaling; ++x)
        {
            size_t addr = x + y * 100;
            if (field_num[addr] > 0)
            {
                field_avg[addr] = field_avg[addr] / field_num[addr];
            }
            field_num[addr] = 0;
        }
    }

    for (size_t i=0; i < tops->points.size(); ++i)
    {
        tf::Vector3 act(tops->points[i].x,tops->points[i].y,tops->points[i].z);
        act = act + shift;
        if ((act.x() > 0) && (act.y() > 0) && (act.y() < 1) && (act.x() < 1))
        {
            int xcoord = act.x() * scaling;
            int ycoord = act.y() * scaling;
            size_t addr = xcoord + ycoord * 100;

            field_sigma[addr] += (act.z() - field_avg[addr]) * (act.z() - field_avg[addr]);
            field_num[addr]++;
        }
    }

    double sigma_max = -10000;
    double sigma_min = 10000;

    for (size_t y = 0; y < scaling; ++y)
    {
        for (size_t x = 0; x < scaling; ++x)
        {
            size_t addr = x + y * 100;
            if (field_num[addr] > 0)
            {
                field_sigma[addr] = sqrt(field_sigma[addr] / field_num[addr]);
                if (field_num[addr] > 20)
                {
                    if (field_sigma[addr]  > sigma_max)
                        sigma_max = field_sigma[addr];
                    if (field_sigma[addr]  < sigma_min)
                        sigma_min = field_sigma[addr];
                }
            }
        }
    }

    boost::shared_ptr<std::vector<int> > indices( new std::vector<int> );

    double delta = .04;

    for (size_t i=0; i < tops->points.size(); ++i)
    {
        tf::Vector3 act_(tops->points[i].x,tops->points[i].y,tops->points[i].z);
        tf::Vector3 act = act_ + shift;
        if ((act.x() > 0) && (act.y() > 0) && (act.y() < 1) && (act.x() < 1))
        {
            int xcoord = act.x() * scaling;
            int ycoord = act.y() * scaling;
            size_t addr = xcoord + ycoord * 100;

            field_idx[addr].push_back(i);
        }
    }

    double min_c = 10000;
    double max_c = -10000;
    int t = 0;
    for (t = 0; t <= 1; t++)
    for (size_t y = 0; y < scaling; ++y)
    {
        for (size_t x = 0; x < scaling; ++x)
        {
            size_t addr = x + y * 100;
            if ((field_idx[addr].size() > 10))
            {
                int id0 = field_idx[addr][0];
                tf::Vector3 pt(tops->points[id0].x,tops->points[id0].y,tops->points[id0].z);
                std::vector<tf::Vector3> evec;
                std::vector<double> eval;
                //if ((pt - tf::Vector3(-1.63,1.33,.86)).length() < 0.05)
                {
                    //std::cout << field_idx[addr].size() << " " << pt.x() << " " << pt.y() << endl;
                    pos_eigen_zcol(field_idx[addr], tops, evec, eval, false); //((pt - tf::Vector3(-1.63,1.33,.86)).length() < 0.05));

                    tf::Vector3 mean(0,0,0);
                    for (std::vector<int>::iterator it = field_idx[addr].begin(); it != field_idx[addr].end(); it ++)
                        mean += pcl_to_zcol(tops->points[*it]);

                    mean = mean / field_idx[addr].size();

                    if ((fabs(evec[0].x()) > 0.0000001) && (fabs(evec[0].y()) > 0.0000001))
                    for (std::vector<int>::iterator it = field_idx[addr].begin(); it != field_idx[addr].end(); it ++)
                    {
                        // project vector on main axis
                        tf::Vector3 cur = pcl_to_zcol(tops->points[*it]) - mean;
                        if (max_c == min_c)
                            max_c += 0.00001;
                        double col = cur.dot(evec[0]);
                        if (t == 0) {
                        if (col > max_c)
                            max_c = col;
                        if (col < min_c)
                            min_c = col;
                        }
                        col = (col - min_c) / (max_c - min_c) * 255;
                        tops->points[*it].r = col;
                        tops->points[*it].g = -col;
                        tops->points[*it].b = 0;
                    }
                }
                    //projected.push_back(pcl_to_zcol(cloud->points[*it]));

                // do some statistics on the points in this bin

                // find the covariance in height + color and use it to find the principle axis

            }
        }
    }

    std::cout << "col limits " << min_c << " " << max_c << std::endl;


            //if (fabs(field[xcoord + ycoord * 100] - act.z()) < 0.0000000001)
            /*
            double dist_to_max = fabs(field[addr] - act.z());
            double dist_to_min = fabs(field_low[addr] - act.z());
            double dist_to_avg = act.z() - field_avg[addr];
            float fac =  dist_to_max / .02;
            //if (dist_to_max == 0)
            //  indices->push_back(i);
            //if (act.z() > field_avg[addr] + 0.01)
            //if (field_sigma[addr] < 0.0025)
            if (act.z() < field_avg[addr] + 0.003)
            {
                tops->points[i].z =  field_avg[addr];
                tops->points[i].r = 255;
                tops->points[i].g = 0;
                tops->points[i].b = 0;
            }

            indices->push_back(i);
            //    std::cout << "si " << field_sigma[addr];
            */

            //if (act.z() > field_avg[addr] + 2 * field_sigma[addr])

            /*if (fabs(act_.x() + 1.63) < 0.02)
                if (fabs(act_.y() - 1.5) < 0.02)
                {
                    std::cout << "field_sigma " << field_sigma[addr] << std::endl;
                    std::cout << "field_avg" << field_avg[addr] << std::endl;
                    std::cout << "field_low " << field_low[addr] << std::endl;
                    std::cout << "field" << field[addr] << std::endl;
                }*/

            //tops->points[i].r = ((field_sigma[addr] - sigma_min) / (sigma_max - sigma_min)) * 255;
            //tops->points[i].g = tops->points[i].r ;
            //tops->points[i].b = tops->points[i].r ;

            /*if (dist_to_max < delta / 3)
            {
                indices->push_back(i);
                cloud->points[i].r =  0;
                cloud->points[i].g =  0;
                cloud->points[i].b =  255;
                field_topvec[addr].push_back(act_);
                //field_topvec[i] += act_;
                //field_topvec_n[i] += 1;
            }

            if ((dist_to_max > 2 * delta / 3) && (dist_to_max < delta))
            {
                indices->push_back(i);
                cloud->points[i].r =  255;
                cloud->points[i].g =  0;
                cloud->points[i].b =  0;
                field_botvec[addr].push_back(act_);
                //field_botvec[i] += act_;
                //field_botvec_n[i] += 1;
            }*/
        //}
    //}

    std::cout << " sigma max " << sigma_max  << " sigma min " << sigma_min << std::endl;

    std::cout << std::endl;

    geometry_msgs::PoseArray parr;
    parr.header.frame_id = msg.header.frame_id;
    parr.header.stamp = ros::Time(0);

    visualization_msgs::MarkerArray marr;
    /*
    for (size_t y = 0; y < scaling; ++y)
    {
        for (size_t x = 0; x < scaling; ++x)
        {
            size_t addr = x + y * 100;
            if ((field_topvec[addr].size() > 10) && (field_botvec[addr].size() > 10))
            {
                tf::Vector3 top(0,0,0);
                tf::Vector3 bot(0,0,0);
                for (std::vector<tf::Vector3>::iterator it = field_topvec[addr].begin(); it != field_topvec[addr].end(); ++it)
                    top += *it;
                top /= field_topvec[addr].size();
                for (std::vector<tf::Vector3>::iterator it = field_botvec[addr].begin(); it != field_botvec[addr].end(); ++it)
                    bot += *it;
                bot /= field_botvec[addr].size();

                double top_at = 0;
                double bot_at = 0;
                double num_at = 0;
                for (std::vector<tf::Vector3>::iterator it = field_topvec[addr].begin(); it != field_topvec[addr].end(); ++it)
                {
                    tf::Vector3 rel = *it - top;
                    if (fabs(rel.x()) > 0.00000000001)
                    {
                        top_at += atan(rel.y()/rel.x());
                        num_at++;
                    }
                }
                for (std::vector<tf::Vector3>::iterator it = field_botvec[addr].begin(); it != field_botvec[addr].end(); ++it)
                {
                    tf::Vector3 rel = *it - bot;
                    if (fabs(rel.x()) > 0.00000000001)
                    {
                        top_at += atan(rel.y()/rel.x());
                        num_at++;
                    }
                }
                top_at /= num_at;

                tf::Vector3 x_axis = top - bot;
                tf::Vector3 y_axis = tf::Vector3(cos(top_at),sin(top_at),0);
                tf::Vector3 z_axis = (x_axis.cross(y_axis)).normalize();
                x_axis = (y_axis.cross(z_axis)).normalize();
                y_axis = (z_axis.cross(x_axis)).normalize();

                if (x_axis.x() != x_axis.x())
                {
                    continue;
                }

                if (top.z() < 0.9)
                    continue;

                //ROS_INFO("x_AXIS %f %f %f len %f", x_axis.x(), x_axis.y(), x_axis.z(), x_axis.length());
                //ROS_INFO("y_AXIS %f %f %f len %f", y_axis.x(), y_axis.y(), y_axis.z(), y_axis.length());
                //ROS_INFO("z_AXIS %f %f %f len %f", z_axis.x(), z_axis.y(), z_axis.z(), z_axis.length());

                btMatrix3x3 rot(x_axis.x(), y_axis.x(), z_axis.x(),
                                x_axis.y(), y_axis.y(), z_axis.y(),
                                x_axis.z(), y_axis.z(), z_axis.z());
                btQuaternion rot_quat;
                rot.getRotation(rot_quat);

                tf::Pose pose;
                pose.setOrigin(bot);
                pose.setRotation(rot_quat);
                geometry_msgs::Pose pose_msg;
                tf::poseTFToMsg(pose,pose_msg);
                parr.poses.push_back(pose_msg);

                addMarker(marr, pose);

                //pose.setOrigin(bot + tf::Vector3(0.005,0.005,0));
                //tf::poseTFToMsg(pose,pose_msg);
                //parr.poses.push_back(pose_msg);

            }
        }
    }
    */

    marker_pub_arr.publish(marr);


    parr_flat_pub.publish(parr);

    //pcl::ExtractIndices<pcl::PointXYZRGB> ei;
    //ei.setInputCloud(tops);
    //ei.setIndices(indices);
    //ei.filter(*z_maxima);

    sensor_msgs::PointCloud2 z_max_msg;
    //pcl::toROSMsg(*z_maxima, z_max_msg);
    pcl::toROSMsg(*tops, z_max_msg);
    z_max_msg.header = msg.header;
    pct_pub.publish(z_max_msg);

}



int main(int argc, char **argv)
{

    ros::init(argc, argv, "mc_graspable", ros::init_options::AnonymousName);

    ros::NodeHandle nh;

    listener_ = new tf::TransformListener();

    pct_pub =  nh.advertise<sensor_msgs::PointCloud2>( "/debug_cloud", 10 , true);
    parr_pub = nh.advertise<geometry_msgs::PoseArray>( "/grasp_poses", 10 , true);
    parr_flat_pub = nh.advertise<geometry_msgs::PoseArray>( "/flat_grasp_poses", 10 , true);
    marker_pub = nh.advertise<visualization_msgs::Marker>( "/grasp_marker", 10 , true);
    marker_pub_arr = nh.advertise<visualization_msgs::MarkerArray>( "/grasp_marker_array", 10 , true);

    ros::Duration(1).sleep();

    ros::Rate rt(30);

    if (atoi(argv[1]) == 1)
    {
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        ros::Time time_stamp;
        //getCloud(cloud,"/map",ros::Time(0), &time_stamp);
        sensor_msgs::PointCloud2 msg;
        getCloud(msg, "/map", ros::Time(0), &time_stamp);
        classify_cloud(msg);
    }

    if (atoi(argv[1]) == 2)
    {
        rosbag::Bag bag;
        bag.open(argv[2], rosbag::bagmode::Read);
        rosbag::View view(bag);

        BOOST_FOREACH(rosbag::MessageInstance const m, view)
        {
            std::string topicname = m.getTopic();
            //ROS_INFO("topic of msg: |%s|", topicname.c_str());
            sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (msg != NULL)
            {
                sensor_msgs::PointCloud2 out_msg = *msg;
                ROS_INFO("GOT THE CLOUD FROM THE BAG");
                out_msg.header.stamp = ros::Time::now();
                pct_pub.publish(out_msg);
                classify_cloud(out_msg);
            }

        }
    }

    if (atoi(argv[1]) == 3)
    {
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        ros::Time time_stamp;
        //getCloud(cloud,"/map",ros::Time(0), &time_stamp);
        sensor_msgs::PointCloud2 msg;
        getCloud(msg, "/map", ros::Time(0), &time_stamp);
        rosbag::Bag bag;
        bag.open(argv[2], rosbag::bagmode::Write);
        bag.write("cloud", ros::Time::now(), msg);
        bag.close();
    }


    if (atoi(argv[1]) == 4)
    {
        rosbag::Bag bag;
        bag.open(argv[2], rosbag::bagmode::Read);
        rosbag::View view(bag);

        BOOST_FOREACH(rosbag::MessageInstance const m, view)
        {
            std::string topicname = m.getTopic();
            //ROS_INFO("topic of msg: |%s|", topicname.c_str());
            sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (msg != NULL)
            {
                sensor_msgs::PointCloud2 out_msg = *msg;
                ROS_INFO("GOT THE CLOUD FROM THE BAG");
                out_msg.header.stamp = ros::Time::now();
                pct_pub.publish(out_msg);
                classify_cloud_low(out_msg);
            }

        }
    }


    ROS_INFO("DONE");

    while (nh.ok())
    {

        rt.sleep();
    }

}
