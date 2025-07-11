import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist

from rclpy.executors import MultiThreadedExecutor
from lifecycle_msgs.msg import TransitionEvent

from nav_msgs.msg import Odometry
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return yaw_z # in radians

class Spin(Node):
    def __init__(self):
        super().__init__('spin')
        # parametry
        self.declare_parameter('spin_times', 1)
        self.declare_parameter('spin_velocity', 1.0)
        self.spin_times    = float(self.get_parameter('spin_times').value)
        self.spin_velocity = float(self.get_parameter('spin_velocity').value)

        # inicjalizacja stanu
        from geometry_msgs.msg import Quaternion
        self.orientation = Quaternion()
        self.old_yaw     = 0.0
        self.shutdown_timer = None
        self.last_yaw       = 0.0      # do śledzenia poprzedniego odczytu
        self.accumulated    = 0.0      # łączny obrócony kąt

        # szukamy topiku odometrii
        for name, types in self.get_topic_names_and_types():
            if any(t == 'nav_msgs/msg/Odometry' for t in types):
                self.get_logger().info(f"Odometry topic found: {name}")
                self.odom_sub = self.create_subscription(
                    Odometry, name, self.odometry_callback, 10
                )
                break  # zamiast return

        # wydawca komend ruchu
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        

        # obliczamy łączny kąt obrotu [rad]
        self.radians_to_spin = self.spin_times * 2 * math.pi
        self.get_logger().info(f"Obliczony kąt obrotu: {self.radians_to_spin:.2f} rad")

        # subskrypcja zdarzeń tranzycji
        self.event_sub = self.create_subscription(
            TransitionEvent,
            '/velocity_smoother/transition_event',
            self.transition_event_callback,
            10
        )

    def odometry_callback(self, msg):
        self.orientation = msg.pose.pose.orientation

    def transition_event_callback(self, msg):
        if msg.goal_state.id == 3:
            start_yaw = euler_from_quaternion(
                self.orientation.x,
                self.orientation.y,
                self.orientation.z,
                self.orientation.w
            )
            self.get_logger().info(f"Start obrotu od yaw={start_yaw:.2f}")
            # inicjalizacja akumulatora
            self.last_yaw    = start_yaw
            self.accumulated = 0.0
            self.timer = self.create_timer(0.05, self.spin)

    def spin(self):
        current_yaw = euler_from_quaternion(
            self.orientation.x,
            self.orientation.y,
            self.orientation.z,
            self.orientation.w
        )
        # obliczamy małą różnicę kąta, z uwzględnieniem zawijania do [-π, π]
        delta = current_yaw - self.last_yaw
        delta = (delta + math.pi) % (2*math.pi) - math.pi

        # akumulujemy
        self.accumulated += delta
        self.last_yaw = current_yaw

        elapsed = abs(self.accumulated)
        self.get_logger().info(f"Elapsed accumulated: {elapsed:.2f} rad")

        if elapsed < abs(self.radians_to_spin):
            msg = Twist()
            msg.angular.z = self.spin_velocity
            self.cmd_vel_pub.publish(msg)
        else:
            # zatrzymanie i shutdown jak wcześniej
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info(f"Obrót zakończony: {elapsed:.2f} rad")
            self.timer.cancel()
            if self.shutdown_timer is None:
                def do_shutdown():
                    self.get_logger().info("Wyłączanie węzła…")
                    rclpy.shutdown()
                self.shutdown_timer = self.create_timer(1.0, do_shutdown)

def shutdown(node_instance):
    node_instance.get_logger().info("Wyłączanie węzła...")
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    spin_robot = Spin()

    executor = MultiThreadedExecutor()
    executor.add_node(spin_robot)
    try:
        executor.spin()
    except KeyboardInterrupt:
        shutdown(spin_robot) 

if __name__ == '__main__':
    main()
