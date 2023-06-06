import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
  ld = LaunchDescription()
  
  # config = os.path.join(
  #   get_package_share_directory('ros2_tutorials'),
  #   'config',
  #   'params.yaml'
  # )
      
  stereo_vo_node = Node(
    package='visual_odometry',
    executable='stereo_vo_node',
    output='screen',
    parameters=[
        {'topicname_image_left': '/camera/infra1/image_rect_raw'},
        {'topicname_image_right': '/camera/infra2/image_rect_raw'},
        {'topicname_pose': '/stereo_vo/pose'},
        {'topicname_trajectory': '/stereo_vo/trajectory'},
        {'topicname_map_points': '/stereo_vo/map_points'},
        {'topicname_debug_image': '/stereo_vo/debug/image'},
        {'directory_intrinsic': '/home/kch/ros2_ws/src/visual_odometry_ros/config/stereo/exp_stereo_ir.yaml'}
    ]
    # parameters = [config]
  )

  ld.add_action(stereo_vo_node)
  return ld