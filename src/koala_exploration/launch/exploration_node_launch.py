from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import yaml
import os

def generate_launch_description():
    # Ścieżki do pakietów
    exploration_dir = FindPackageShare('koala_exploration')
    nav2_bringup_dir = FindPackageShare('koala_navigation')
    slam_dir = FindPackageShare('koala_slam')

    # LaunchConfiguration
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Argumenty
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([exploration_dir, 'params', 'explore.yaml']),
        description='Ścieżka do pliku parametrów ROS2'
    )

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Użycie zegara symulacji (Gazebo) jeśli true'
    )

    # Include SLAM launch
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([slam_dir, 'launch', 'slam_toolbox_mapping.launch.py'])
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # Include Nav2 bringup
    
    nav2_bringup_launch = TimerAction(
                period=5.0,
                actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([nav2_bringup_dir, 'launch', 'navigation_final_launch.py'])
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file,
        }.items(),
    )])

    # Funkcja do dynamicznego włączania węzłów na podstawie parametrów
    def configure_nodes(context):
        # Pobierz ścieżkę do pliku parametrów
        params_file_path = context.launch_configurations.get('params_file', '')
        
        # Jeśli to nie jest pełna ścieżka, użyj domyślnej
        if not os.path.isabs(params_file_path):
            # Domyślna ścieżka - trzeba będzie ją dostosować do rzeczywistej lokalizacji
            params_file_path = '/home/jakub/dev_magisterka/src/koala_exploration/params/explore.yaml'
        
        nodes = []
        
        # Wczytaj parametry z pliku YAML
        try:
            with open(params_file_path, 'r') as file:
                config = yaml.safe_load(file)
                explorer_params = config.get('explorer', {}).get('ros__parameters', {})
                # Pobieranie parametru
                enable_wfd = explorer_params.get('USE_WFD_SOURCE_BENCH', False)
                enable_ffd = explorer_params.get('USE_FFD_SOURCE_BENCH', False)

                if enable_wfd:
                    nodes.append(Node(
                            package='koala_exploration',
                            executable='WFD_node',
                            name='WFD',
                            parameters=[params_file],
                            output='screen'
                    ))

                if enable_ffd:
                    nodes.append(Node(
                            package='koala_exploration',
                            executable='FFD_node',
                            name='FFD',
                            parameters=[params_file],
                            output='screen'
                        )
                    )
                    
        except Exception as e:
            print(f"Błąd wczytywania pliku parametrów: {e}")
            # W przypadku błędu, uruchom domyślnie FFD (zgodnie z YAML)
            Node(
                package='koala_exploration',
                executable='FFD_node',
                name='FFD',
                parameters=[params_file],
                output='screen')

        return nodes

    # Węzeł eksploracji
    delayed_explore_koala = TimerAction(
        period=5.0,
        actions=[Node(
            package='koala_exploration',
            executable='explorer_node',
            name='explorer',
            parameters=[params_file],
            output='screen'
        )]
    )

    return LaunchDescription([
        declare_params_file_cmd,
        declare_use_sim_time_cmd,
        slam_launch,
        nav2_bringup_launch,
        delayed_explore_koala,
        OpaqueFunction(function=configure_nodes),
    ])