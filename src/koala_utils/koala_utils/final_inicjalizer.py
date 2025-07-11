import rclpy
from rclpy.node import Node
import yaml
from ament_index_python.packages import get_package_share_directory
import os
from rclpy.executors import MultiThreadedExecutor
from koala_intefaces.srv import GlobalLocalization
from collections import Counter
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, Pose, Point, Quaternion
from std_msgs.msg import Header
from lifecycle_msgs.srv import GetState
import math

class InitPose(Node):
    def __init__(self):
        super().__init__('init_pose')

        # Liczba prób lokalizacji (x-krotnie)
        self.declare_parameter('num_attempts', 5)
        self.num_attempts = self.get_parameter('num_attempts').value

        # Klient usługi GlobalLocalization
        self.client = self.create_client(GlobalLocalization, 'global_localization_srv')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for GlobalLocalization service...')

        # Klient serwisu AMCL get_state
        self.amcl_state_client = self.create_client(GetState, '/amcl/get_state')

        # Publisher dla inicjalnej pozycji
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/initialpose', 
            10
        )

        # Wyzwalacz inicjalizacji
        self.call_localization()

    def check_amcl_state(self):
        """
        Sprawdza stan węzła AMCL
        Zwraca: stan AMCL jako string lub None w przypadku błędu
        """
        if not self.amcl_state_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('AMCL get_state service not available')
            return None
        
        request = GetState.Request()
        future = self.amcl_state_client.call_async(request)
        
        # Czekaj na odpowiedź
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        if future.result() is not None:
            response = future.result()
            state_label = response.current_state.label
            self.get_logger().info(f'AMCL current state: {state_label}')
            return state_label
        else:
            self.get_logger().error('Failed to get AMCL state')
            return None

    def call_localization(self):
        
        poses = []
        for i in range(self.num_attempts):
            self.get_logger().info(f"Proba: {i}")
            req = GlobalLocalization.Request()
            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            if future.result() is None:
                self.get_logger().error(f'Localization attempt {i+1} failed')
                continue

            res = future.result()
            x_r = round(res.x, 2)
            y_r = round(res.y, 2)
            t_r = round(res.theta, 2)
            poses.append((x_r, y_r, t_r))
            self.get_logger().info(f'Attempt {i+1}: x={x_r}, y={y_r}, theta={t_r}')

        if not poses:
            self.get_logger().error('All localization attempts failed. Cannot initialize pose.')
            return

        # Wybór pozycji modalnej (najczęściej powtarzanej)
        mode_pose, count = Counter(poses).most_common(1)[0]
        x_final, y_final, theta_final = mode_pose
        self.get_logger().info(
            f'Chosen mode pose (repeated {count} times): x={x_final}, y={y_final}, theta={theta_final}'
        )

        # Delegacja do metody użytkownika
        result = self.run_init_pose(x_final, y_final, theta_final)
        

        # Zakończenie działania
        self.destroy_node()
        rclpy.shutdown()

    def publish_initial_pose(self, x, y, theta):
        """
        Publikuje inicjalną pozycję robota przez temat /initialpose
        """
        msg = PoseWithCovarianceStamped()
        
        # Header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        # Pose
        msg.pose = PoseWithCovariance()
        msg.pose.pose = Pose()
        
        # Position
        msg.pose.pose.position = Point()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = 0.0
        
        # Orientation (konwersja z theta na quaternion)
        msg.pose.pose.orientation = Quaternion()
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        
        # Covariance - wszystkie wartości na 0 (jak requested)
        msg.pose.covariance = [0.0] * 36  # 6x6 matrix = 36 elements
        
        # Publikuj wiadomość
        self.initial_pose_pub.publish(msg)
        self.get_logger().info(f"Published initial pose: x={x}, y={y}, theta={theta}")
        
        # Czekaj chwilę, żeby wiadomość została dostarczona
        import time
        time.sleep(0.5)

    def run_init_pose(self, x, y, theta):
        try:
            # Sprawdź stan AMCL przed rozpoczęciem lokalizacji
            self.get_logger().info('Checking AMCL state before localization...')
            amcl_state = self.check_amcl_state()
            
            if amcl_state != 'active':
                self.get_logger().warning(f'AMCL is not active (current state: {amcl_state}). Localization may not work properly.')
                if amcl_state is None:
                    self.get_logger().error('Cannot proceed without AMCL state information.')
                    return False
            else:
                self.get_logger().info('AMCL is active. Localization will proceed.')
                self.publish_initial_pose(x, y, theta)
                return True

        except Exception as e:
            self.get_logger().error(f"Błąd podczas inicjalizacji pozycji: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False

def shutdown(node):
    node.destroy_node()
    rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    pose_initializer = InitPose()
    executor.add_node(pose_initializer)

    try:
        executor.spin()
    except KeyboardInterrupt:
        shutdown(pose_initializer)

