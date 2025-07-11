import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    # Ścieżka do katalogu pakietu "koala_navigation"
    koala_nav_dir = get_package_share_directory('koala_navigation')

    # Konfiguracje uruchomieniowe
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    navigation_on = LaunchConfiguration('navigation_on')
    num_attempts = LaunchConfiguration('num_attempts')

    # Deklaracja argumentów launch
    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(koala_nav_dir, 'maps', 'Sala', 'map.yaml'),
        description='Pełna ścieżka do pliku mapy'
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Czy używać zegara symulacji (Gazebo)'
    )
    
    declare_params_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(koala_nav_dir, 'params', 'nav2_params_final.yaml'),
        description='Pełna ścieżka do pliku parametrów ROS2'
    )

    declare_navigation_on_cmd = DeclareLaunchArgument(
        'navigation_on',
        default_value='true',
        description='Czy wlaczyc nawigacje'
    )
    
    declare_num_attempts_cmd = DeclareLaunchArgument(
        'num_attempts',
        default_value='3',
        description='Liczba prób inicjalizacji pozycji'
    )

    # Definicja węzła global_localizer_service
    global_localizer_service_cmd = Node(
        package='global_localizer',
        executable='global_localizer',
        name='global_localizer',
        parameters=[{"map_yaml": map_yaml_file}],
        output='screen'
    )

    # Definicja węzła inicjalizacji pozycji
    init_pose_cmd = Node(
        package='koala_utils',
        executable='init_pose',
        name='pose_initializer',
        parameters=[{"map_url": map_yaml_file}, 
                    {"use_sim": use_sim_time},
                    {"num_attempts": num_attempts}],
        output='screen'
    )

    # Załączenie launch file dla nawigacji
    navi_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(koala_nav_dir, 'launch', 'generic_launch.py')
        ),
        condition=IfCondition(navigation_on),
        launch_arguments={
            'map': map_yaml_file,
            'use_sim_time': use_sim_time,
            'params_file': params_file
        }.items()
    )



    # Utworzenie LaunchDescription oraz dodanie akcji
    ld = LaunchDescription()

    ld.add_action(LogInfo(msg=['Ścieżka do mapy: ', map_yaml_file]))
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(global_localizer_service_cmd)
    ld.add_action(declare_navigation_on_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_num_attempts_cmd)
    ld.add_action(declare_params_cmd)
    ld.add_action(navi_launch)
    # Event handler: Po uruchomieniu global_localizer, uruchom init_pose_cmd
    ld.add_action(RegisterEventHandler(
        OnProcessStart(
            target_action=global_localizer_service_cmd,
            on_start=[
                init_pose_cmd,
                LogInfo(msg=['Global localizer uruchomiony. Inicjalizacja pozycji...']),
            ]
        )
    ))

    return ld
