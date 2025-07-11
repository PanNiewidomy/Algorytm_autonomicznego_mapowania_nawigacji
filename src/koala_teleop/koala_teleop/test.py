#!/usr/bin/env python3
"""
Robot Movement Timer with Odometry
===================================
Program liczy czas ruchu robota używając danych z odometrii ROS2.
"""

import time
import threading
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from typing import Optional

class RobotTimer:
    def __init__(self, stop_threshold: float = 5.0, velocity_threshold: float = 0.01):
        self.stop_threshold = stop_threshold  # Czas zatrzymania w sekundach
        self.velocity_threshold = velocity_threshold  # Próg prędkości do uznania za ruch
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.is_moving = False
        self.stop_timer = None
        self.total_movement_time = 0.0
        self.current_velocity = 0.0
        
    def update_velocity(self, linear_vel: float, angular_vel: float):
        """Aktualizuj prędkość robota (kombinacja liniowej i kątowej)"""
        # Oblicz całkowitą prędkość jako kombinację liniowej i kątowej
        self.current_velocity = math.sqrt(linear_vel**2 + (angular_vel * 0.5)**2)
        
        # Sprawdź czy robot się porusza
        if self.current_velocity > self.velocity_threshold:
            if not self.is_moving:
                self.robot_started_moving()
        else:
            if self.is_moving:
                self.robot_stopped_moving()
        
    def robot_started_moving(self):
        """Wywołaj gdy robot zaczyna się ruszać"""
        if not self.is_moving:
            self.start_time = time.time()
            self.is_moving = True
            self.stop_time = None
            
            # Anuluj timer zatrzymania jeśli istnieje
            if self.stop_timer:
                self.stop_timer.cancel()
                self.stop_timer = None
                
            print(f"🚀 Robot zaczął się ruszać o {time.strftime('%H:%M:%S', time.localtime(self.start_time))}")
            print(f"   Prędkość: {self.current_velocity:.3f} m/s")
    
    def robot_stopped_moving(self):
        """Wywołaj gdy robot zatrzymuje się"""
        if self.is_moving:
            print(f"⏸️  Robot zatrzymał się o {time.strftime('%H:%M:%S')}")
            print(f"   Prędkość: {self.current_velocity:.3f} m/s")
            
            # Uruchom timer - jeśli robot nie ruszy przez threshold sekund, zakończ pomiar
            self.stop_timer = threading.Timer(self.stop_threshold, self._finalize_measurement)
            self.stop_timer.start()
    
    def _finalize_measurement(self):
        """Finalizuj pomiar po zatrzymaniu na wymagany czas"""
        if self.is_moving and self.start_time:
            self.stop_time = time.time()
            raw_time = self.stop_time - self.start_time
            self.total_movement_time = raw_time - self.stop_threshold
            self.is_moving = False
            
            print(f"🏁 Robot zatrzymał się na {self.stop_threshold} sekund")
            print(f"📊 Wyniki pomiaru:")
            print(f"   • Czas całkowity: {raw_time:.2f} sekund")
            print(f"   • Czas zatrzymania: {self.stop_threshold:.2f} sekund")
            print(f"   • Czas rzeczywistego ruchu: {self.total_movement_time:.2f} sekund")
            print(f"   • Start: {time.strftime('%H:%M:%S', time.localtime(self.start_time))}")
            print(f"   • Stop: {time.strftime('%H:%M:%S', time.localtime(self.stop_time))}")
    
    def get_movement_time(self) -> float:
        """Zwraca czas ruchu (bez czasu zatrzymania)"""
        return self.total_movement_time
    
    def reset(self):
        """Resetuj timer"""
        if self.stop_timer:
            self.stop_timer.cancel()
        self.start_time = None
        self.stop_time = None
        self.is_moving = False
        self.stop_timer = None
        self.total_movement_time = 0.0
        print("🔄 Timer zresetowany")


class RobotTimerNode(Node):
    def __init__(self):
        super().__init__('robot_timer_node')
        
        # Inicjalizuj timer
        self.timer = RobotTimer(stop_threshold=5.0, velocity_threshold=0.01)
        
        # Subskrypcje
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',  # Standardowy topic odometrii
            self.odometry_callback,
            10
        )
        
        # Opcjonalnie subskrypcja cmd_vel dla dodatkowych informacji
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Timer do regularnego raportowania stanu
        self.status_timer = self.create_timer(2.0, self.status_callback)
        
        self.get_logger().info("🤖 Robot Timer Node uruchomiony")
        self.get_logger().info(f"   Próg prędkości: {self.timer.velocity_threshold} m/s")
        self.get_logger().info(f"   Próg zatrzymania: {self.timer.stop_threshold} s")
        
        # Dodatkowe informacje
        self.last_cmd_vel = None
        self.odom_received = False
        
    def odometry_callback(self, msg: Odometry):
        """Callback dla danych odometrii"""
        if not self.odom_received:
            self.get_logger().info("📡 Otrzymuję dane z odometrii")
            self.odom_received = True
            
        # Pobierz prędkości z odometrii
        linear_vel = math.sqrt(
            msg.twist.twist.linear.x**2 + 
            msg.twist.twist.linear.y**2
        )
        angular_vel = abs(msg.twist.twist.angular.z)
        
        # Aktualizuj timer
        self.timer.update_velocity(linear_vel, angular_vel)
        
    def cmd_vel_callback(self, msg: Twist):
        """Callback dla komend prędkości (opcjonalne)"""
        self.last_cmd_vel = msg
        
    def status_callback(self):
        """Regularny raport stanu"""
        if self.odom_received:
            status = "🏃 RUCH" if self.timer.is_moving else "⏹️  STOP"
            self.get_logger().info(
                f"{status} | Prędkość: {self.timer.current_velocity:.3f} m/s | "
                f"Czas ruchu: {self.timer.get_movement_time():.1f}s"
            )
        else:
            self.get_logger().warning("⚠️  Brak danych z odometrii")
    
    def get_results(self):
        """Zwróć wyniki pomiaru"""
        return {
            'movement_time': self.timer.get_movement_time(),
            'is_moving': self.timer.is_moving,
            'current_velocity': self.timer.current_velocity,
            'start_time': self.timer.start_time,
            'stop_time': self.timer.stop_time
        }


def main(args=None):
    rclpy.init(args=args)
    
    node = None
    try:
        node = RobotTimerNode()
        
        print("🚀 Robot Timer z odometrią uruchomiony")
        print("   Naciśnij Ctrl+C aby zakończyć")
        print("   Robot automatycznie wykrywa ruch na podstawie odometrii")
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n👋 Zatrzymywanie programu...")
        
        # Wyświetl końcowe wyniki
        if node and node.timer.total_movement_time > 0:
            results = node.get_results()
            print(f"\n✅ Końcowe wyniki:")
            print(f"   Czas rzeczywistego ruchu: {results['movement_time']:.2f} sekund")
        
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


# Funkcja testowa bez ROS2
def test_without_ros():
    """Test bez ROS2 - symulacja danych odometrii"""
    timer = RobotTimer(stop_threshold=3.0, velocity_threshold=0.01)
    
    print("� Test bez ROS2 - symulacja odometrii")
    
    # Symuluj dane odometrii
    test_velocities = [
        (0.0, 0.0, 1),    # Start - brak ruchu
        (0.5, 0.0, 3),    # Ruch do przodu
        (0.0, 1.0, 2),    # Obrót
        (0.3, 0.5, 4),    # Kombinowany ruch
        (0.0, 0.0, 5),    # Stop
    ]
    
    for linear, angular, duration in test_velocities:
        print(f"📊 Symulacja: v_lin={linear}, v_ang={angular}, czas={duration}s")
        
        for _ in range(duration * 10):  # 10 Hz
            timer.update_velocity(linear, angular)
            time.sleep(0.1)
    
    print(f"\n✅ Test zakończony. Czas ruchu: {timer.get_movement_time():.2f}s")


if __name__ == "__main__":
    import sys
    
    if '--test' in sys.argv:
        test_without_ros()
    else:
        main()