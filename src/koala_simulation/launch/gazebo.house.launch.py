from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import launch_ros
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Package names\ n    
    package_description = 'koala_description'
    package_simulation = 'koala_simulation'
    ekf_package_description = 'koala_perception'
    world_package_description = 'aws_robomaker_small_house_world'
    rviz_config_file = 'config2.rviz'

    # Declare input arguments
    gui_run = LaunchConfiguration('gui')
    rviz_run = LaunchConfiguration('rviz')
    teleop_run = LaunchConfiguration('teleop_run')

    declare_gui_arg = DeclareLaunchArgument(
        'gui',
        default_value='false',
        description='Enable world GUI'
    )
    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='false',
        description='Enable RViz'
    )
    declare_teleop_arg = DeclareLaunchArgument(
        'teleop_run',
        default_value='false',
        description='Enable teleop'
    )

    # Load robot description
    load_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(package_description), 'launch'),
            '/load_description.launch.py'
        ])
    )

    # Simulation world (small house)
    small_house = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(world_package_description),
                'launch',
                'small_house.launch.py'
            )
        ),
        launch_arguments={'gui': gui_run}.items()
    )

    # EKF state estimation
    ekf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(ekf_package_description), 'launch'),
            '/ekf.launch.py'
        ])
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'koala_bot', '-z', '0.1'],
        output='screen'
    )

    # RViz visualization
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(get_package_share_directory(package_simulation), 'config', rviz_config_file)],
        condition=IfCondition(rviz_run)
    )

    # Teleoperation node
    teleop = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop',
        prefix='xterm -e',
        output='screen',
        condition=IfCondition(teleop_run)
    )

    return LaunchDescription([
        declare_gui_arg,
        declare_rviz_arg,
        declare_teleop_arg,
        launch_ros.actions.SetParameter(name='use_sim_time', value=True),
        load_description,
        small_house,
        spawn_entity,
        ekf,
        rviz2,
        teleop
    ])
