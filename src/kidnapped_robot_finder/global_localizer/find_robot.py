import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import numpy as np
import cv2
from global_localizer import kidnap_solver as ks
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import yaml
import os
from koala_intefaces.srv import GlobalLocalization

class LaserScanFilter(Node):

    def __init__(self):
        super().__init__('global_localizer_node')

        self.declare_parameter('map_yaml', '')
        map_yaml_path = self.get_parameter('map_yaml').value
        if not map_yaml_path:
            self.get_logger().error("Nie podano parametru map_yaml! Użyj: ros2 run global_localizer global_localizer --ros-args --parameter  map_yaml:=/path/to/map.yaml")
            rclpy.shutdown()
            return

        self.max_range = 8.0  # max range of lidar in meters
        self.resolution = 0.05  # 5 cm per pixel

        self.min_distance = None
        self.scan_image = None
        self.map_image = None
        self.config_data = None

        self.scan_topic = "/scan" # default scan topic
        self.map_file_path = ""

        self.max_iterations = 30
        self.stop_search_threshold_f1 = 50
        
        self.lidar_max_range = 8.0 
        self.map_resolution = 0.05 # m/px

        self.load_parameters()
        self.load_map_from_yaml(map_yaml_path)
        self.robot_in_map_meters = None
        self.robot_angle_in_map = None
        

        self.image_size = int((2 * self.max_range) / self.resolution)  # image width and height in pixels
        self.origin_offset = int(self.max_range / self.resolution)  # origin offset in pixels
        
        self.subscription = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10
        )

        self.srv = self.create_service(GlobalLocalization, 'global_localization_srv', self.global_localization_callback)
        self.get_logger().info('Global Localization Service is ready.')


    def scan_callback(self, msg: LaserScan):
        # Initialize a black image
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        min_distance = math.inf
        # Convert lidar points to pixels and draw on the image
        for i, range_val in enumerate(msg.ranges):
            if 0 < range_val < self.max_range:  # Ignore invalid or out-of-range values
                angle = msg.angle_min + i * msg.angle_increment
                # Convert polar coordinates (range, angle) to Cartesian (x, y)
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                
                # Convert from meters to pixels
                px = int((x / self.resolution) + self.origin_offset)
                py = int((y / self.resolution) + self.origin_offset)
                min_distance = min(min_distance, range_val)
                # Draw the point on the image as a white circle with radius 1
                cv2.circle(image, (px, py), radius=1, color=255, thickness=-1)

        #print(f"Min distance: {min_distance}")

        self.scan_image = image.copy()
        self.min_distance = min_distance

    def load_config_file(self):
        package_share_directory = get_package_share_directory('global_localizer')
        yaml_file_path = Path(package_share_directory) / 'config' / 'config.yaml'
        try:
            with open(yaml_file_path, 'r') as file:
                self.config_data = yaml.safe_load(file)
        except Exception as e:
            self.get_logger().error(f"Error reading YAML file: {e}")
            exit()

    def load_parameters(self):
        self.load_config_file()
        self.scan_topic = self.config_data['scan_topic']

        self.max_iterations = self.config_data['max_iterations']
        self.stop_search_threshold_f1 = self.config_data['stop_search_threshold_f1']

        self.lidar_max_range = self.config_data['lidar_max_range']
        self.map_resolution = self.config_data['map_resolution']

        self.threshold_distance = self.config_data['threshold_distance']

        self.get_logger().info(f"Scan topic: {self.scan_topic}")
        self.get_logger().info(f"Map file path: {self.map_file_path}")
        self.get_logger().info(f"Max iterations: {self.max_iterations}")
        self.get_logger().info(f"Stop search threshold: {self.stop_search_threshold_f1}")
        self.get_logger().info(f"Lidar max range: {self.lidar_max_range}")
        self.get_logger().info(f"Map resolution: {self.map_resolution}")

    def load_map_file(self):
        map_file_path = self.map_file_path
        self.get_logger().info(f"Map file path: {map_file_path}")

        if Path(map_file_path).exists():
            map_image = cv2.imread(map_file_path, cv2.IMREAD_GRAYSCALE)
            self.map_image = map_image
        else:
            self.get_logger().error(f"Map file not found: {map_file_path}")
            exit()

    def load_map_from_yaml(self, map_yaml_path):
        """
        Wczytujemy plik map.yaml, wyciągamy z niego:
         - ścieżkę do pliku obrazka,
         - resolution,
         - origin.
        Ładujemy też samą mapę (plik .pgm czy .png).
        """
        try:
            with open(map_yaml_path, 'r') as file:
                map_data = yaml.safe_load(file)

            # Zakładamy, że plik map.yaml ma klucze:
            #  image, resolution, origin (lista 3 liczb: [x, y, theta])
            self.map_file_path= os.path.join(os.path.dirname(map_yaml_path),map_data['image'])
            self.map_resolution = float(map_data['resolution'])
            self.map_origin = map_data['origin']  # np. [-7.47, -5.27, 0.0]

            self.get_logger().info(f"Wczytano map.yaml: {map_yaml_path}")
            self.get_logger().info(f"Map image file: {self.map_file_path}")
            self.get_logger().info(f"Resolution: {self.map_resolution}")
            self.get_logger().info(f"Origin: {self.map_origin}")

            # Wczytujemy obraz (zakładamy grayscale)
            if not Path(self.map_file_path).exists():
                self.get_logger().error(f"Map image not found: {self.map_file_path}")
                rclpy.shutdown()
                return
            self.load_map_file()

        except Exception as e:
            self.get_logger().error(f"Error reading map.yaml: {e}")
            rclpy.shutdown()

    def global_localization_callback(self, request, response):
        self.get_logger().info('Global Localization Service triggered.')
        
        # Czekaj na dane ze skanu, maksymalnie przez N sekund
        max_wait_time = 10.0  # maksymalny czas oczekiwania w sekundach
        wait_increment = 0.1  # sprawdzaj co 100ms
        total_waited = 0.0
        
        while self.min_distance is None and total_waited < max_wait_time:
            self.get_logger().info(f'Waiting for laser scan data... ({total_waited:.1f}s)')
            # Użyj time.sleep zamiast rclpy.spin_once() by nie blokować innych callbacków
            import time
            time.sleep(wait_increment)
            total_waited += wait_increment
        
        if self.min_distance is not None:
            self.get_logger().info('Found laser-scan data. Matching in progress....')
            
            # Solve the kidnap problem
            robot_in_map_meters, yaw_img = ks.solve_kidnap(self.scan_image, 
                            self.map_image, 
                            self.min_distance, 
                            map_resolution = self.map_resolution,
                            max_iterations = self.max_iterations, 
                            stop_search_threshold = self.stop_search_threshold_f1, 
                            lidar_range = self.lidar_max_range,
                            map_origin = self.map_origin,
                            threshold_distance = self.threshold_distance)
            # 2) odwrócenie osi Y (Y↓→Y↑) – tak jak miałeś
            yaw_map = -yaw_img

            # 3) dodaj obrót mapy (jeśli origin[2] ≠ 0)
            yaw_map += self.map_origin[2]

            # 4) skompensuj flip o 180°
            yaw_map += math.pi

            # 5) znormalizuj do zakresu [-π, π]
            yaw_map = math.atan2(math.sin(yaw_map), math.cos(yaw_map))

            # 6) wypełnij odpowiedź
            response.x     = robot_in_map_meters[0]
            response.y     = robot_in_map_meters[1]
            response.theta = yaw_map
            
        else:
            self.get_logger().error('No laser-scan data received within timeout period. Please make sure laser scans are available and retry.')
        
        return response

def main(args=None):
    print("\n\n** Starting the localization service!!**\n\n")
    rclpy.init(args=args)
    laser_scan_filter = LaserScanFilter()
    rclpy.spin(laser_scan_filter)
    laser_scan_filter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
