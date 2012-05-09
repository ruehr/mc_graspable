#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <mc_graspable/FindGraspablesAction.h>

int main (int argc, char **argv)
{
  ros::init(argc, argv, "test_find_graspables");

  // create the action client
  // true causes the client to spin its own thread
  actionlib::SimpleActionClient<mc_graspable::FindGraspablesAction> ac("mc_graspable", true);

  ROS_INFO("Waiting for action server to start.");
  // wait for the action server to start
  ac.waitForServer(); //will wait for infinite time

  ROS_INFO("Action server started, sending goal.");
  // send a goal to the action
  mc_graspable::FindGraspablesGoal goal;

  goal.cloud_topic = "/kinect/rgb/points";
  goal.frame_id = "/map";
  goal.aabb_min.x = -2.5;
  goal.aabb_min.y = 1;
  goal.aabb_min.z = .5;
  goal.aabb_max.x = -1.5;
  goal.aabb_max.y = 2;
  goal.aabb_max.z = 1.5;
  goal.delta = 0.02;
  goal.scaling = 20;
  goal.pitch_limit = 0.4;
  goal.thickness = 0.04;
  ac.sendGoal(goal);

  //wait for the action to return
  bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

  if (finished_before_timeout)
  {
    actionlib::SimpleClientGoalState state = ac.getState();
    ROS_INFO("Action finished: %s",state.toString().c_str());
  }
  else
    ROS_INFO("Action did not finish before the time out.");

  //exit
  return 0;
}
