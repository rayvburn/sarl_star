#!/usr/bin/python2.7
# Author: Keyu Li <kyli@link.cuhk.edu.hk>

from __future__ import division
import math
import logging
import os
import torch
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
import configparser
import gym
import tf
import tf2_ros
import tf2_geometry_msgs
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import ObservableState, FullState, JointState
import rospy
from geometry_msgs.msg import Point, Vector3, Twist, Pose, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from std_msgs.msg import Int32, ColorRGBA
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from people_msgs.msg import Person, People
from visualization_msgs.msg import Marker, MarkerArray


HUMAN_RADIUS = 0.3
ROBOT_RADIUS = 0.3
ROBOT_V_PREF = 0.5
DISCOMFORT_DIST = 0.5
FAKE_HUMAN_PX = -1.7
FAKE_HUMAN_PY = 14.3
TIME_LIMIT = 1000
GOAL_TOLERANCE = 0.2

def add(v1, v2):
    return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

def transform_pose(pose_stamped, transform):
    return tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)

def find_transform(source_frame, target_frame="map", stamp=rospy.Time(), timeout=rospy.Duration(1.0)):
    if source_frame == target_frame:
        transform = TransformStamped()
        transform.header.frame_id = source_frame
        transform.header.stamp = rospy.Time.now()
        transform.transform.rotation.w = 1.0 # no rotation
        return True, transform
    try:
        transform = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            stamp,
            timeout
        )
        return True, transform
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr("Could not find a TF: " + str(e))
        return False, TransformStamped()

class Robot(object):
    def __init__(self, v_pref, radius, goal_tolerance):
        self.v_pref = v_pref
        self.radius = radius
        self.goal_tolerance = goal_tolerance
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None

    def set(self, px, py, gx, gy, vx, vy, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy

    def reached_destination(self):
        return np.linalg.norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.goal_tolerance
        #      || (position - goal position) ||


class Human(object):
    def __init__(self, px, py, vx, vy):
        self.radius = HUMAN_RADIUS
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

class RobotAction(object):
    def __init__(self):
        self.Is_lg_Received = False
        self.IsAMCLReceived = False
        self.IsObReceived = False
        self.Is_gc_Received = False
        self.getStartPoint = False
        self.Is_lg_Reached = False
        self.Is_gg_Reached = False
        self.received_gx = None
        self.received_gy = None
        self.px = None
        self.py = None
        self.vx = None
        self.vy = None
        self.gx = None
        self.gy = None
        self.v_pref = None
        self.theta = None
        self.humans = None
        self.ob = None
        self.state = None
        self.cmd_vel = Twist()
        self.plan_counter = 0
        self.num_pos = 0
        self.num_lg = 0
        self.start_px = None
        self.start_py = None

        # ROS subscribers
        self.robot_pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.update_robot_pos)
        self.robot_odom_sub = rospy.Subscriber('/odom', Odometry, self.robot_vel_on_map_calculator)
        self.people_sub = rospy.Subscriber('/people', People, self.update_humans)
        self.goal_sub = rospy.Subscriber('/local_goal', PoseStamped, self.get_goal_on_map)
        self.global_costmap_sub = rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, self.get_gc)
        # ROS publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=1)
        self.goal_marker_pub = rospy.Publisher('/goal_marker', Marker, queue_size=1)
        self.action_marker_pub = rospy.Publisher('/action_marker', Marker, queue_size=1)
        self.trajectory_marker_pub = rospy.Publisher('/trajectory_marker', Marker, queue_size=1)
        self.vehicle_marker_pub = rospy.Publisher('/vehicle_marker', Marker, queue_size=1)

    def update_robot_pos(self, msg):
        self.num_pos += 1
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.theta = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))  # bounded by [-pi, pi]
        if not self.getStartPoint:
            rospy.loginfo("Start point is:(%s,%s)" % (self.px,self.py))
            self.getStartPoint = True
        self.visualize_trajectory(position, orientation)
        self.IsAMCLReceived = True

    def robot_vel_on_map_calculator(self, msg):
        vel_linear = msg.twist.twist.linear
        status, transform = find_transform(source_frame='base_footprint', target_frame='map')
        if not status:
            rospy.logerr("Could not find a TF - robot velocity won't be computed.")
            return
        # rotate vector 'vel_linear' by quaternion 'rot'
        q1 = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]
        q2 = list()
        q2.append(vel_linear.x)
        q2.append(vel_linear.y)
        q2.append(vel_linear.z)
        q2.append(0.0)
        output_vel = tf.transformations.quaternion_multiply(
            tf.transformations.quaternion_multiply(q1, q2),
            tf.transformations.quaternion_conjugate(q1)
        )[:3]
        self.vx = output_vel[0]
        self.vy = output_vel[1]

    def update_humans(self, msg):
        # observable state: px,py,vx,vy,radius
        self.humans = list()
        self.ob = list()

        transform_needed = not "map" in msg.header.frame_id
        if transform_needed: # either "map" or "/map"
            # must transform poses to the shared frame
            status, transform = find_transform(
                source_frame=msg.header.frame_id,
                target_frame='map',
                stamp=msg.header.stamp,
                timeout=rospy.Duration(4.0)
            )
            if not status:
                rospy.logerr("Could not find a TF - humans list won't be updated.")
                return

        for p in msg.people:
            if not transform_needed:
                human = Human(p.position.x, p.position.y, p.velocity.x, p.velocity.y)
            else:
                # transform pose
                pose_input = PoseStamped()
                pose_input.header.frame_id = msg.header.frame_id
                pose_input.header.stamp = msg.header.stamp
                pose_input.pose.position = p.position
                pose_input.pose.orientation.w = 1.0 # arbitrary as the orientation is not available here
                pose_transformed = transform_pose(pose_input, transform)

                # transform velocity to the global coordinate system (only rotation needed)
                vel_input = PoseStamped()
                vel_input.header.frame_id = msg.header.frame_id
                vel_input.header.stamp = msg.header.stamp
                quat = tf.transformations.quaternion_from_euler(0.0, 0.0, math.atan2(p.velocity.y, p.velocity.x))
                vel_input.pose.orientation.x = quat[0]
                vel_input.pose.orientation.y = quat[1]
                vel_input.pose.orientation.z = quat[2]
                vel_input.pose.orientation.w = quat[3]
                vel_transformed = transform_pose(vel_input, transform)
                rpy = tf.transformations.euler_from_quaternion(
                    [vel_transformed.pose.orientation.x,
                    vel_transformed.pose.orientation.y,
                    vel_transformed.pose.orientation.z,
                    vel_transformed.pose.orientation.w]
                )
                vel_dir = rpy[2]
                # base vector represented as [1.0, 0.0], scaled according to the current speed
                vvec = [math.sqrt(p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y), 0.0]
                vel_transformed_x = (vvec[0] * math.cos(vel_dir) - vvec[1] * math.sin(vel_dir))
                vel_transformed_y = (vvec[0] * math.sin(vel_dir) + vvec[1] * math.cos(vel_dir))

                human = Human(
                    pose_transformed.pose.position.x,
                    pose_transformed.pose.position.y,
                    vel_transformed_x,
                    vel_transformed_y
                )
            # dist = np.linalg.norm(np.array([self.px,self.py])-np.array([human.px,human.py]))
            self.humans.append(human)
        for human in self.humans:
            self.ob.append(human.get_observable_state())
        self.IsObReceived = True

    def get_goal_on_map(self, msg):
        transform_needed = not "map" in msg.header.frame_id
        if transform_needed: # either "map" or "/map"
            # must transform poses to the shared frame
            status, transform = find_transform(source_frame=msg.header.frame_id, target_frame="map")
            if not status:
                rospy.logerr("Robot goal won't be updated")
                return
            tfmsg = transform_pose(msg, transform)
        else:
            tfmsg = msg
        self.received_gx = tfmsg.pose.position.x
        self.received_gy = tfmsg.pose.position.y
        self.Is_lg_Received = True

    def get_gc(self, msg):
        if not self.Is_gc_Received:
            policy.gc = msg.data
            policy.gc_resolution = msg.info.resolution
            policy.gc_width = msg.info.width
            policy.gc_ox = msg.info.origin.position.x
            policy.gc_oy = msg.info.origin.position.y
            # print(policy.gc_resolution, policy.gc_width, policy.gc_ox, policy.gc_oy)
            print("************ Global costmap is received. **************")
            self.Is_gc_Received = True

    def visualize_goal(self):
        # red cube for local goals
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal"
        marker.id = self.num_lg
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position.x = self.gx
        marker.pose.position.y = self.gy
        marker.pose.position.z = 0.2
        marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
        marker.color = ColorRGBA(r=1.0, a=1.0)
        marker.lifetime = rospy.Duration()
        self.goal_marker_pub.publish(marker)

    def visualize_trajectory(self, position, orientation):
        # Purple track for robot trajectory over time
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = '/map'
        marker.ns = 'robot'
        marker.id = self.num_pos
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.pose.position = position
        marker.pose.orientation = orientation
        marker.scale = Vector3(x=0.1, y=0.1, z=0.1)
        marker.color = ColorRGBA(r=0.5, b=0.8, a=1.0)
        marker.lifetime = rospy.Duration()
        self.trajectory_marker_pub.publish(marker)

    def visualize_action(self):
        robot_pos = Point(x=self.px, y=self.py, z=0)
        next_theta = self.theta + self.cmd_vel.angular.z
        next_vx = self.cmd_vel.linear.x * np.cos(next_theta)
        next_vy = self.cmd_vel.linear.x * np.sin(next_theta)
        action = Vector3(x=next_vx, y=next_vy, z=0)
        # green arrow for action (command velocity)
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "/map"
        marker.ns = "action"
        marker.id = 0
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.points = [robot_pos, add(robot_pos, action)]
        marker.scale = Vector3(x=0.1, y=0.3, z=0)
        marker.color = ColorRGBA(g=1.0, a=1.0)
        marker.lifetime = rospy.Duration(0.5)
        self.action_marker_pub.publish(marker)

    def planner(self):
        # update robot
        robot.set(self.px, self.py, self.gx, self.gy, self.vx, self.vy, self.theta)

        # compute command velocity
        if robot.reached_destination():
            self.cmd_vel.linear.x = 0
            self.cmd_vel.linear.y = 0
            self.cmd_vel.linear.z = 0
            self.cmd_vel.angular.x = 0
            self.cmd_vel.angular.y = 0
            self.cmd_vel.angular.z = 0
            self.Is_lg_Reached = True
            if self.gx == self.received_gx and self.gy == self.received_gy:
                self.Is_gg_Reached = True
        else:
            """
            self state: FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta)
            ob:[ObservableState(px1, py1, vx1, vy1, radius1),
                ObservableState(px1, py1, vx1, vy1, radius1),
                   .......                    
                ObservableState(pxn, pyn, vxn, vyn, radiusn)]
            """
            if len(self.ob)==0:
                self.ob = [ObservableState(FAKE_HUMAN_PX, FAKE_HUMAN_PY, 0, 0, HUMAN_RADIUS)]

            self.state = JointState(robot.get_full_state(), self.ob)
            action = policy.predict(self.state)  # max_action
            self.cmd_vel.linear.x = action.v
            self.cmd_vel.linear.y = 0
            self.cmd_vel.linear.z = 0
            self.cmd_vel.angular.x = 0
            self.cmd_vel.angular.y = 0
            self.cmd_vel.angular.z = action.r

        ########### for debug ##########
        # dist_to_goal = np.linalg.norm(np.array(robot.get_position()) - np.array(robot.get_goal_position()))
        # if self.plan_counter % 10 == 0:
        #     rospy.loginfo("robot position:(%s,%s)" % (self.px, self.py))
        #     rospy.loginfo("Distance to goal is %s" % dist_to_goal)
        #     rospy.loginfo("self state:\n %s" % self.state.self_state)
        #     for i in range(len(self.state.human_states)):
        #         rospy.loginfo("human %s :\n %s" % (i+1, self.state.human_states[i]))
        #     rospy.loginfo("%s-th action is planned: \n v: %s m/s \n r: %s rad/s"
        #                   % (self.plan_counter, self.cmd_vel.linear.x, self.cmd_vel.angular.z))


        # publish command velocity
        self.cmd_vel_pub.publish(self.cmd_vel)
        self.plan_counter += 1
        self.visualize_action()


def publish_computation_time(timestamp_start, timestamp_finish):
    computation_time = float(timestamp_finish - timestamp_start)
    msg = Float64MultiArray()
    msg.data = [float(timestamp_start), float(computation_time)]

    msg.layout.data_offset = 0
    dim_stamp = MultiArrayDimension()
    dim_stamp.label = 'stamp'
    dim_stamp.size = 1
    dim_stamp.stride = 1
    msg.layout.dim.append(dim_stamp)
    dim_ct = MultiArrayDimension()
    dim_ct.label = 'computation_time'
    dim_ct.size = 1
    dim_ct.stride = 1
    msg.layout.dim.append(dim_ct)
    comp_time_pub.publish(msg)


if __name__ == '__main__':
    begin_travel = False
    # set file dirs
    this_dirname = os.path.dirname(__file__)
    model_dir = os.path.join(this_dirname, '../CrowdNav/crowd_nav/data/output/')
    env_config_file = os.path.join(model_dir, 'env.config')
    policy_config_file = os.path.join(model_dir, 'policy.config')
    if os.path.exists(os.path.join(model_dir, 'resumed_rl_model.pth')):
        model_weights = os.path.join(model_dir, 'resumed_rl_model.pth')
    else:
        model_weights = os.path.join(model_dir, 'rl_model.pth')

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, x%(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cpu")
    logging.info('Using device: %s', device)

    # configure RL policy
    policy = 'sarl'
    phase = 'test'
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    env.discomfort_dist = DISCOMFORT_DIST
    policy = policy_factory[policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    policy.with_costmap = True
    # use constant velocity model to predict next state
    policy.query_env = False  
    policy.get_model().load_state_dict(torch.load(model_weights))
    policy.set_phase(phase)
    policy.set_device(device)
    policy.set_env(env)
    policy.time_step = 0.25
    policy.gc = []

    rospy.init_node('sarl_star_node', anonymous=True)
    robot_v_pref = rospy.get_param('~vel_pref', ROBOT_V_PREF)
    robot_radius = rospy.get_param('~robot_radius', ROBOT_RADIUS)
    goal_tolerance = rospy.get_param('~goal_tolerance', GOAL_TOLERANCE)
    robot = Robot(v_pref=robot_v_pref, radius=robot_radius, goal_tolerance=goal_tolerance)
    comp_time_pub = rospy.Publisher('~computation_time', Float64MultiArray, queue_size=1)

    try:
        rate = rospy.Rate(4)  # 4Hz, time_step=0.25
        tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) # tf buffer length
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        robot_act = RobotAction()

        while not rospy.is_shutdown():
            if robot_act.Is_gg_Reached:
                finish_travel_time = rospy.get_time()
                t = finish_travel_time - begin_travel_time
                rospy.loginfo("Goal is reached. Travel time: %s s." % t)
                break

            # wait for msgs of goal, AMCL and ob
            if robot_act.Is_lg_Received and robot_act.IsAMCLReceived and robot_act.IsObReceived:

                # travel time
                if not begin_travel:
                    begin_travel_time = rospy.get_time()
                    begin_travel = True

                # update local goal (gx,gy)
                robot_act.gx = robot_act.received_gx
                robot_act.gy = robot_act.received_gy
                robot_act.num_lg += 1
                robot_act.visualize_goal()

                # count the computation time of the next control command
                computation_time_start = rospy.Time.now()
                robot_act.planner()
                computation_time_finish = rospy.Time.now()
                publish_computation_time(computation_time_start.to_sec(), computation_time_finish.to_sec())

                finish_travel_time = rospy.get_time()
                t = finish_travel_time - begin_travel_time
                if t > TIME_LIMIT:
                    rospy.loginfo("Timeout. Travel time: %s s." % t)
                    break
            rate.sleep()

    except rospy.ROSInterruptException, e:
        raise e



