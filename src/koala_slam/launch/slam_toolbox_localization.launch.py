from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

import os

def generate_launch_description():

    start_localization_slam_toolbox_node = Node(
        parameters=[
            os.path.join(
            get_package_share_directory('koala_slam'),
            'params','mapper_params_localization.yaml'),
            {"map_file_name": get_package_share_directory("koala_slam") + '/res/map'},
        ],
        package='slam_toolbox',
        executable='localization_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
    )
    return LaunchDescription([
        start_localization_slam_toolbox_node
    ])
