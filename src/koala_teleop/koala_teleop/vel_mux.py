import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class VelMux(Node):
    def __init__(self):
        super().__init__('vel_mux')
        # Ustawienie limitu czasu na 100 ms (0.1 s)
        self.timeout = 0.1

        # Subskrypcje tematów
        self.high_priority_sub = self.create_subscription(
            Twist,
            '/vel_high_priority',
            self.high_priority_callback,
            10
        )
        self.low_priority_sub = self.create_subscription(
            Twist,
            '/vel_low_priority',
            self.low_priority_callback,
            10
        )

        # Publisher dla /cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer ustawiony na 25 Hz
        timer_period = 1.0 / 25.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Zmienne do przechowywania ostatnio odebranych wiadomości oraz czasu ich nadejścia
        self.last_high_msg = None
        self.last_high_time = None
        self.last_low_msg = None
        self.last_low_time = None

    def high_priority_callback(self, msg: Twist):
        # Zapisz wiadomość i czas jej nadejścia
        self.last_high_msg = msg
        self.last_high_time = self.get_clock().now()

    def low_priority_callback(self, msg: Twist):
        self.last_low_msg = msg
        self.last_low_time = self.get_clock().now()

    def timer_callback(self):
        now = self.get_clock().now()

        # Obliczamy, ile sekund upłynęło od odebrania ostatnich wiadomości
        def time_diff_in_seconds(msg_time):
            if msg_time is None:
                return float('inf')  # Jeśli nie ma czasu, traktuj jak „nieskończony” odstęp
            return (now - msg_time).nanoseconds / 1e9

        high_age = time_diff_in_seconds(self.last_high_time)
        low_age = time_diff_in_seconds(self.last_low_time)

        # Sprawdzamy, czy high/low priority msg jest wciąż świeża (mniej niż self.timeout s)
        high_fresh = (high_age < self.timeout)
        low_fresh = (low_age < self.timeout)

        # Domyślnie nadajemy zerową prędkość
        twist = Twist()

        if high_fresh:
            # Jeśli high priority jest świeża, używamy jej
            twist = self.last_high_msg
        elif low_fresh:
            # W przeciwnym razie, jeśli low priority jest świeża, używamy jej
            twist = self.last_low_msg
        # Jeśli żaden z sygnałów nie jest świeży, zostawiamy zerową prędkość

        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = VelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
