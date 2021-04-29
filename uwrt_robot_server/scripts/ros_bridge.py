#!/usr/bin/env python

import numpy as np
from scipy.spatial.transform import Rotation as R

from threading import Event

import rospy

import tf2_ros
import geometry_msgs.msg

from gazebo_msgs.msg import ModelState, LinkState, ContactsState
from gazebo_msgs.srv import GetModelState, SetModelState, GetLinkState, SetLinkState
from gazebo_msgs.srv import SetModelConfiguration, SetModelConfigurationRequest

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header
from std_srvs.srv import Empty

from visualization_msgs.msg import Marker

from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

#######################
#######################

class UWRTRosBridge:

    def __init__(self,  real_robot=False):

        # Event is clear while initialization or set_state is going on
        self.reset = Event()
        self.reset.clear()
        self.get_state_event = Event()
        self.get_state_event.set()

        self.real_robot = real_robot

        # GAZEBO SERVICE CALLS
        self.model_name = 'uw_mars_rover'
        self.end_effector_frame = 'uw_mars_rover::hex_key'
        self.reference_frame = 'uw_mars_rover::base_link'

        # Joint States
        # 0 - base_link_joint
        # 1 - elbow
        # 2 - hex_joint
        # 3 - servo_joint
        # 4 - shoulder
        # 5 - turntable_to_chassis
        # 6 - wheel_left_back_to_rocker
        # 7 - wheel_left_front_to_rocker
        # 8 - wheel_right_back_to_rocker
        # 9 - wheel_right_front_to_rocker
        # 10 - wrist_rotate
        # 11 - wrist_up_down
        self.uwrt_arm_joint_state_idxs = np.array([5, 4, 1, 11, 10])
        self.num_arm_joints = len(self.uwrt_arm_joint_state_idxs)
        rospy.Subscriber("joint_states", JointState, self.UWRTJointStateCallback)

        self.gazebo_observation = {
            'goal': {
                'desired_key_pose_in_world_frame': {
                    "position": [0] * 3,
                    "orientation": [0] * 4,
                },
            },
            'arm': {
                "position": [0] * self.num_arm_joints,
                "velocity": [0] * self.num_arm_joints,
                "effort": [0] * self.num_arm_joints,
                'allen_key_pose_in_world_frame': {
                    "position": [0] * 3,
                    "orientation": [0] * 4,
                },
            },
        }

        # Target RViz Marker publisher
        self.desired_key_pose_pub  = rospy.Publisher('desired_key', Marker, queue_size=10)
        self.keyboard_pose_pub     = rospy.Publisher('keyboard', Marker, queue_size=10)

        # joint_trajectory_command_handler publisher
        self.arm_cmd_pub = rospy.Publisher('env_arm_command', Float64MultiArray, queue_size=1)

        # Robot control rate
        self.sleep_time = (1.0/rospy.get_param("~action_cycle_rate")) - 0.01
        self.control_period = rospy.Duration.from_sec(self.sleep_time)

        # TF2 Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform_broadcaster = tf2_ros.TransformBroadcaster()

        # Desired Key
        self.desired_key = geometry_msgs.msg.TransformStamped()
        self.desired_key.header.frame_id = 'base_link'
        self.desired_key.child_frame_id = 'desired_key'

    def __gazebo_observation_to_rs_state(self):
        """ Can't send dict's through Robot Server """
        rs_state = []
        rs_state.extend(list(self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['position']))
        rs_state.extend(list(self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['orientation']))
        rs_state.extend(list(self.gazebo_observation['arm']['position']))
        rs_state.extend(list(self.gazebo_observation['arm']['velocity']))
        rs_state.extend(list(self.gazebo_observation['arm']['effort']))
        rs_state.extend(list(self.gazebo_observation['arm']['allen_key_pose_in_world_frame']['position']))
        rs_state.extend(list(self.gazebo_observation['arm']['allen_key_pose_in_world_frame']['orientation']))
        return rs_state

    def __update_gazebo_observation(self, rs_state):

        self.gazebo_observation = {
            'goal': {
                'desired_key_pose_in_world_frame': {
                    "position":  rs_state[0:3],
                    "orientation": rs_state[3:7],
                },
            },
            'arm': {
                "position": rs_state[7:12],
                "velocity": rs_state[12:17],
                "effort": rs_state[17:22],
                'allen_key_pose_in_world_frame': {
                    "position": rs_state[22:25],
                    "orientation": rs_state[25:29],
                },
            },
        }

        #######################
        #######################

        self.desired_key.header.stamp = rospy.Time.now()
        self.desired_key.transform.translation.x = self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['position'][0]
        self.desired_key.transform.translation.y = self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['position'][1]
        self.desired_key.transform.translation.z = self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['position'][2]
        self.desired_key.transform.rotation.x = self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['orientation'][0]
        self.desired_key.transform.rotation.y = self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['orientation'][1]
        self.desired_key.transform.rotation.z = self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['orientation'][2]
        self.desired_key.transform.rotation.w = self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['orientation'][3]

        self.transform_broadcaster.sendTransform(self.desired_key)

    def UWRTJointStateCallback(self, data):
        if self.get_state_event.is_set():
            # joint states
            self.gazebo_observation['arm']['position'] = np.array(data.position)[self.uwrt_arm_joint_state_idxs]
            self.gazebo_observation['arm']['velocity'] = np.array(data.velocity)[self.uwrt_arm_joint_state_idxs]
            self.gazebo_observation['arm']['effort'] = np.array(data.effort)[self.uwrt_arm_joint_state_idxs]
            # claw link state
            self.get_link_state(link_name=self.end_effector_frame, reference_frame=self.reference_frame)

    def get_link_state(self, link_name='', reference_frame=''):
        """ Getting End Effector Position and Orienation """

        rospy.wait_for_service('/gazebo/get_link_state')
        try:
            link_state_srv = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            link_coordinates = link_state_srv(link_name, reference_frame).link_state

            x = link_coordinates.pose.position.x
            y = link_coordinates.pose.position.y
            z = link_coordinates.pose.position.z
            position = np.array([x, y, z])

            x = link_coordinates.pose.orientation.x
            y = link_coordinates.pose.orientation.y
            z = link_coordinates.pose.orientation.z
            w = link_coordinates.pose.orientation.w
            orientation = np.array([x, y, z, w])

            self.gazebo_observation['arm']['allen_key_pose_in_world_frame']['position'] = position
            self.gazebo_observation['arm']['allen_key_pose_in_world_frame']['orientation'] = orientation

        except rospy.ServiceException as e:
            print("Service call failed:" + e)

    def get_home_joint_position(self):
        # for testing
        home_joint_position = [0, 1, 1, 1.75, 0]

        # TODO: randomize start config
        # home_joint_position = np.array([np.random.uniform(-np.pi/8, np.pi/8),          # turntable_to_chassis
        #                                 np.random.uniform(0.65, 0.85),                 # shoulder
        #                                 np.random.uniform(0.35, 0.65),                 # elbow
        #                                 np.random.uniform(1.35, 1.65),                 # wrist_up_down
        #                                 np.random.uniform(-np.pi / 8, np.pi / 8)       # wrist_rotate
        #                                 ])

        print("Reseting UWRT Arm Joint Positions: {}".format(home_joint_position))
        return home_joint_position

    def get_state(self):
        # Clear get_state_event Event
        self.get_state_event.clear()

        # set get_state_event Event
        self.get_state_event.set()

        return robot_server_pb2.State(state=self.__gazebo_observation_to_rs_state(), success=1)

    def set_state(self, state_msg):
        # Set environment state
        rs_state = state_msg.state

        # Clear reset Event
        self.reset.clear()

        # update gazebo state
        self.__update_gazebo_observation(rs_state)

        # Publish Target Marker
        self.publish_keyboard()

        reset_steps = int(45.0/self.sleep_time)
        joint_home_positions = self.get_home_joint_position()
        for i in range(reset_steps):
            self.publish_env_arm_cmd(joint_home_positions)

        # set reset Event
        self.reset.set()

        return 1

    def publish_env_arm_cmd(self, pos_cmd):
        """ publish 'pos_cmd' action to ROS Controller """
        msg = Float64MultiArray()
        msg.data = pos_cmd
        self.arm_cmd_pub.publish(msg)
        rospy.sleep(self.control_period)

    def publish_keyboard(self):

        position = self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['position']
        orientation = self.gazebo_observation['goal']['desired_key_pose_in_world_frame']['orientation']

        # desired key: 15.5mm x 15mm size
        DESIRED_KEY = Marker()
        DESIRED_KEY.header.stamp = rospy.Time.now()
        DESIRED_KEY.header.frame_id = '/base_link'
        DESIRED_KEY.type = DESIRED_KEY.CUBE # desired_key_pose_in_world_frame as a 'cube'
        DESIRED_KEY.action = DESIRED_KEY.ADD
        DESIRED_KEY.scale.x = 0.01
        DESIRED_KEY.scale.y = 0.0155
        DESIRED_KEY.scale.z = 0.015
        DESIRED_KEY.color.a = 0.85
        DESIRED_KEY.color.b = 0.5
        DESIRED_KEY.color.g = 0.0
        DESIRED_KEY.color.r = 0.5
        DESIRED_KEY.pose.position.x = position[0]
        DESIRED_KEY.pose.position.y = position[1]
        DESIRED_KEY.pose.position.z = position[2]
        DESIRED_KEY.pose.orientation.x = orientation[0]
        DESIRED_KEY.pose.orientation.y = orientation[1]
        DESIRED_KEY.pose.orientation.z = orientation[2]
        DESIRED_KEY.pose.orientation.w = orientation[3]
        self.desired_key_pose_pub.publish(DESIRED_KEY)

        # todo: foramlize keyboard pose
        # x: 0.825
        # y: -0.25 to 0.25
        # z: 0.675 to 0.825
        KEYBOARD = Marker()
        KEYBOARD.header.stamp = rospy.Time.now()
        KEYBOARD.header.frame_id = '/base_link'
        KEYBOARD.type = KEYBOARD.CUBE  # desired_key_pose_in_world_frame as a 'cube'
        KEYBOARD.action = KEYBOARD.ADD
        KEYBOARD.scale.x = 0.01
        KEYBOARD.scale.y = 0.25
        KEYBOARD.scale.z = 0.075
        KEYBOARD.color.a = 0.85
        KEYBOARD.color.b = 0.0
        KEYBOARD.color.g = 0.0
        KEYBOARD.color.r = 0.0
        KEYBOARD.pose.position.x = 0.825
        KEYBOARD.pose.position.y = 0
        KEYBOARD.pose.position.z = 0.75
        KEYBOARD.pose.orientation.x = 0
        KEYBOARD.pose.orientation.y = 0
        KEYBOARD.pose.orientation.z = 0
        KEYBOARD.pose.orientation.w = 1
        self.keyboard_pose_pub.publish(KEYBOARD)